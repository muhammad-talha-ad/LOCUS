"""CUDA-graphed branch passes over a cached prefix (inference only, fast mode).

A branch pass at batch 1 is ~60 ms of kernel launches for ~12 ms of GPU work (28 layers x ~15 kernels, plus
DynamicCache bookkeeping and a deepcopy of the prefix cache). Here each pass is one graph replay of the same HF
decoder-layer modules (same weights, same SDPA kernel family), on static buffers:

  prefix K/V   copied once per prefill into [R, kv_heads, P, d], P fixed (padded keys masked out)
  branches     right-padded to an Lb bucket, rows padded to an R bucket (padding rows repeat row 0)
  mask         additive [R, 1, Lb, P + Lb]: prefix validity + causal within the branch (what create_causal_mask builds)

The prefix cache is read, never written, so no copy is needed between passes. Graphs are captured lazily per
(model, R, Lb) bucket and share one memory pool.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

R_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
L_BUCKETS = (16, 32, 48, 64, 96, 128, 192, 256)  # 192+: Ref-L4 expressions (up to ~160 tokens)


class _PrefixKV:
    """Stands in for the Cache object inside attention: static prefix keys/values followed by the branch's own."""

    def __init__(self, k, v):
        self.k, self.v = k, v

    def update(self, k, v, layer_idx, *args, **kwargs):
        return torch.cat([self.k[layer_idx], k], 2), torch.cat([self.v[layer_idx], v], 2)


def _bucket(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"{n} exceeds the largest bucket {buckets[-1]}")


class GraphedBranches:
    def __init__(self, vlms: dict, layers: list[int], max_prefix: int, max_images: int = 8):
        """vlms: {key: plain Qwen3-VL ...ForConditionalGeneration} sharing one prefix (e.g. base / merged LoRA);
        layers: hidden_states indices to return (0 = embeddings, n = final normed state, as HF output_hidden_states)."""
        self.texts = {k: v.model.language_model for k, v in vlms.items()}
        text = next(iter(self.texts.values()))
        self.layers, self.P = sorted(set(layers)), max_prefix
        cfg = text.config
        self.n = cfg.num_hidden_layers
        kvh, hd = cfg.num_key_value_heads, getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.dtype, self.dev = text.embed_tokens.weight.dtype, text.embed_tokens.weight.device
        self.Kp = [torch.zeros(max_images, kvh, self.P, hd, dtype=self.dtype, device=self.dev) for _ in range(self.n)]
        self.Vp = [torch.zeros_like(k) for k in self.Kp]
        self.pmask = torch.zeros(max_images, self.P, dtype=torch.bool, device=self.dev)
        self.graphs, self.pool = {}, None

    # -- prefix ---------------------------------------------------------------
    def load_prefix(self, cache, prefix_mask: torch.Tensor):
        """Copy a prefill DynamicCache ([B, kvh, Lp, d] per layer) and its 2-D mask into the static buffers."""
        B, Lp = prefix_mask.shape
        assert Lp <= self.P and B <= self.pmask.shape[0], f"prefix {B}x{Lp} exceeds {tuple(self.pmask.shape)}"
        for i, layer in enumerate(cache.layers):
            self.Kp[i][:B, :, :Lp].copy_(layer.keys)
            self.Vp[i][:B, :, :Lp].copy_(layer.values)
        self.pmask.zero_()
        self.pmask[:B, :Lp] = prefix_mask.bool()

    # -- graphs ---------------------------------------------------------------
    def _graph(self, which, rb, lb):
        key = (which, rb, lb)
        text = self.texts[which]
        if key in self.graphs:
            return self.graphs[key]
        s = SimpleNamespace(ids=torch.zeros(rb, lb, dtype=torch.long, device=self.dev),
                            pos=torch.zeros(3, rb, lb, dtype=torch.long, device=self.dev),
                            mask=torch.zeros(rb, 1, lb, self.P + lb, dtype=self.dtype, device=self.dev),
                            rows=torch.zeros(rb, dtype=torch.long, device=self.dev))
        s.out = {}

        def run():
            kv = _PrefixKV([k[s.rows] for k in self.Kp], [v[s.rows] for v in self.Vp])
            h = text.embed_tokens(s.ids)
            pe = text.rotary_emb(h, s.pos)
            for i, layer in enumerate(text.layers):
                if i in self.layers:
                    s.out[i] = h
                h = layer(h, attention_mask=s.mask, position_ids=s.pos[0], past_key_values=kv, position_embeddings=pe)
            h = text.norm(h)
            s.out[self.n] = h

        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st), torch.no_grad():
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool), torch.no_grad():
            run()
        self.pool = g.pool()
        s.graph = g
        self.graphs[key] = s
        return s

    @torch.no_grad()
    def __call__(self, which, start, branch_ids, row_image):
        """Same contract as BoxVLM.branch_pass: returns .last_hidden_state [R, Lb, d] and .hidden_states {i: [R, Lb, d]}."""
        R, Lb = branch_ids.shape
        rb, lb = _bucket(R, R_BUCKETS), _bucket(Lb, L_BUCKETS)
        s = self._graph(which, rb, lb)
        s.ids.fill_(0)
        s.ids[:R, :Lb] = branch_ids
        s.ids[R:] = s.ids[:1]
        s.rows.zero_()
        s.rows[:R] = row_image
        pos = start[s.rows][:, None] + torch.arange(lb, device=self.dev)[None]
        s.pos.copy_(pos[None].expand(3, -1, -1))
        neg = torch.finfo(self.dtype).min
        causal = torch.ones(lb, lb, dtype=torch.bool, device=self.dev).tril()
        allowed = torch.cat([self.pmask[s.rows][:, None, :].expand(-1, lb, -1), causal[None].expand(rb, -1, -1)], 2)
        s.mask.copy_(torch.where(allowed, 0.0, neg)[:, None].to(self.dtype))
        s.graph.replay()
        hs = {i: s.out[i][:R, :Lb].clone() for i in s.out}
        return SimpleNamespace(last_hidden_state=hs[self.n], hidden_states=hs)


class _RecordKV:
    """Stands in for the Cache object during a prefill: keeps each layer's keys/values, attends to the prefix only."""

    def __init__(self, n):
        self.k, self.v = [None] * n, [None] * n

    def update(self, k, v, layer_idx, *args, **kwargs):
        self.k[layer_idx], self.v[layer_idx] = k, v
        return k, v


class GraphedPrefill:
    """CUDA-graphed [image][prefix] prefill at batch 1 (latency path, fast mode).

    The eager HF prefill is ~6k kernel launches for ~29 ms of GPU work at 512², and syncs on the host in the vision
    tower (per-layer `lengths.tolist()` for SDPA), the position-embedding interpolation and `get_rope_index`. Everything
    that depends only on the image grid (interpolation taps, vision rotary positions, cu_seqlens kept on the host, 3-D
    text positions, image-token indices) is computed once per (grid, prompt length) and the rest is one graph replay of
    the same HF modules: vision tower, masked_scatter as an index copy, decoder layers with DeepStack adds, final norm.
    The prefix prompt is fixed, so the token ids are a function of the grid. Graphs share one memory pool; outputs are
    valid until the next replay (load_prefix copies them out)."""

    def __init__(self, vlm, layers: list[int]):
        self.model = vlm.model
        self.visual, self.text = self.model.visual, self.model.language_model
        self.layers, self.n = sorted(set(layers)), self.text.config.num_hidden_layers
        self.image_token_id = vlm.config.image_token_id
        self.graphs, self.pool = {}, None

    def _graph(self, pi):
        from transformers.vision_utils import (get_vision_cu_seqlens, get_vision_interpolation_indices_and_weights,
                                               get_vision_position_ids)

        ids, grid = pi["input_ids"], pi["image_grid_thw"]
        key = (tuple(grid.flatten().tolist()), ids.shape[1])
        if key in self.graphs:
            return self.graphs[key]
        assert ids.shape[0] == 1 and bool(pi["attention_mask"].all()), "batch-1, unpadded prefill only"
        vis = self.visual
        ii, iw = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=vis.num_grid_per_side, mode=vis.interpolation_mode,
            align_corners=vis.interpolation_align_corners, spatial_merge_size=vis.config.spatial_merge_size)
        pos, _ = self.model.get_rope_index(ids, image_grid_thw=grid, attention_mask=pi["attention_mask"],
                                           mm_token_type_ids=pi["mm_token_type_ids"])
        s = SimpleNamespace(pix=pi["pixel_values"].clone(), grid=grid.clone(), ids=ids.clone(), pos=pos,
                            ii=ii, iw=iw, vpos=get_vision_position_ids(grid, vis.spatial_merge_size),
                            cu=get_vision_cu_seqlens(grid).cpu(),  # host copy: SDPA splits by lengths.tolist()
                            img=(ids[0] == self.image_token_id).nonzero()[:, 0], out={}, key=key)

        def run():
            v = vis(s.pix, grid_thw=s.grid, cu_seqlens=s.cu, position_ids=s.vpos, interp_indices=s.ii,
                    interp_weights=s.iw)
            h = self.text.embed_tokens(s.ids)
            emb = v.pooler_output
            emb = torch.cat(emb, 0) if isinstance(emb, (list, tuple)) else emb
            h = h.index_put((torch.zeros_like(s.img), s.img), emb.to(h.dtype))
            pe = self.text.rotary_emb(h, s.pos)
            kv = _RecordKV(self.n)
            for i, layer in enumerate(self.text.layers):
                if i in self.layers:
                    s.out[i] = h
                h = layer(h, attention_mask=None, position_ids=None, past_key_values=kv, position_embeddings=pe)
                if i < len(v.deepstack_features):
                    h = h.clone()
                    h[0, s.img] = h[0, s.img] + v.deepstack_features[i].to(h.dtype)
            s.out[self.n] = self.text.norm(h)
            s.kv = kv

        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st), torch.no_grad():
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool), torch.no_grad():
            run()
        self.pool = g.pool()
        s.graph = g
        self.graphs[key] = s
        return s

    @torch.no_grad()
    def __call__(self, pi):
        """pi: processor output for one [image][prefix] row (no padding). Returns .hidden_states {i: [1, L, d]},
        .past_key_values (.layers[i].keys/.values, as DynamicCache), .position_ids [3, 1, L] 
        .image_index (image-token positions) and .key ((t, h, w) grid, prompt length), all graph-owned."""
        s = self._graph(pi)
        s.pix.copy_(pi["pixel_values"])
        s.ids.copy_(pi["input_ids"])
        s.graph.replay()
        layers = [SimpleNamespace(keys=k, values=v) for k, v in zip(s.kv.k, s.kv.v)]
        return SimpleNamespace(hidden_states=s.out, past_key_values=SimpleNamespace(layers=layers), position_ids=s.pos,
                               image_index=s.img, key=s.key)


class GraphedJag:
    """The GraphedPrefill recipe applied to Jag (JagRef: Qwen3.5 backbone + box head on the last token), batch 1, so the
    baseline gets the same engineering as our pipeline in latency comparisons. Jag's prompt contains the expression,
    so graphs are keyed by (grid, prompt length); token ids are copied in on every call. Grid-only quantities are
    precomputed as in GraphedPrefill; masks are None, which is what HF builds for one unpadded row."""

    def __init__(self, jag):
        self.jag, self.bb = jag, jag.backbone
        self.image_token_id = self.bb.config.image_token_id
        self.graphs, self.pool = {}, None

    def _graph(self, inputs):
        from transformers.vision_utils import (get_vision_cu_seqlens, get_vision_interpolation_indices_and_weights,
                                               get_vision_position_ids)

        ids, grid = inputs["input_ids"], inputs["image_grid_thw"]
        key = (tuple(grid.flatten().tolist()), ids.shape[1])
        if key in self.graphs:
            return self.graphs[key]
        assert ids.shape[0] == 1 and bool(inputs["attention_mask"].all()), "batch-1, unpadded only"
        vis, text = self.bb.visual, self.bb.language_model
        ii, iw = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=vis.num_grid_per_side, mode=vis.interpolation_mode,
            align_corners=vis.interpolation_align_corners, spatial_merge_size=vis.config.spatial_merge_size)
        pos = self.bb.compute_3d_position_ids(input_ids=ids, image_grid_thw=grid, video_grid_thw=None,
                                              inputs_embeds=None, attention_mask=inputs["attention_mask"],
                                              past_key_values=None, mm_token_type_ids=inputs["mm_token_type_ids"])
        s = SimpleNamespace(pix=inputs["pixel_values"].clone(), grid=grid.clone(), ids=ids.clone(), pos=pos,
                            ii=ii, iw=iw, vpos=get_vision_position_ids(grid, vis.spatial_merge_size),
                            cu=get_vision_cu_seqlens(grid).cpu(),
                            img=(ids[0] == self.image_token_id).nonzero()[:, 0])

        def run():
            v = vis(s.pix.type(vis.dtype), grid_thw=s.grid, cu_seqlens=s.cu, position_ids=s.vpos,
                    interp_indices=s.ii, interp_weights=s.iw)
            h = text.embed_tokens(s.ids)
            h = h.index_put((torch.zeros_like(s.img), s.img), v.pooler_output.to(h.dtype))
            hid = text(input_ids=None, position_ids=s.pos, inputs_embeds=h, use_cache=False,
                       attention_mask={"full_attention": None, "linear_attention": None}).last_hidden_state
            with torch.autocast("cuda", enabled=False):
                s.out = self.jag.bbox_head(hid[:, -1].float())

        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st), torch.no_grad():
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool), torch.no_grad():
            run()
        self.pool = g.pool()
        s.graph = g
        self.graphs[key] = s
        return s

    @torch.no_grad()
    def __call__(self, inputs) -> torch.Tensor:
        """Same contract as JagRef.forward for one unpadded row: normalised cxcywh [1, 4] (fp32, graph-owned)."""
        s = self._graph(inputs)
        s.pix.copy_(inputs["pixel_values"])
        s.ids.copy_(inputs["input_ids"])
        s.graph.replay()
        return s.out


def _capture(run, pool=None):
    """Two warmup runs on a side stream, then one capture into `pool` (None = a private pool); returns the graph."""
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st), torch.no_grad():
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool), torch.no_grad():
        run()
    return g


class GraphedHybrid:
    """Batch-1 CUDA-graphed prefill and branch passes for hybrid backbones (Qwen3.5: Gated DeltaNet + full attention),
    whose linear-attention conv/recurrent states do not fit GraphedBranches' padded-KV scheme. Used for latency.

    Shapes are exact, never bucketed, so every replay runs the same kernels as the eager path (bitwise identical):
      prefill  keyed by (grid, prompt length): vision with grid-only quantities precomputed (as GraphedPrefill), the
               text model on a DynamicCache created inside the graph (static addresses), masks precomputed eagerly
               with HF's own builders.
      branch   keyed by (prefill key, model, branch length), one row: the eager branch_pass on a fork of the
               graph-owned prefix cache (fork + batch_select copies the states, so the prefix is never written).
    All graphs share one memory pool: every graph's outputs stay referenced (self.prefills / self.branches), so a later
    capture only reuses earlier graphs' temporaries, and replays are sequential on one stream.
    Serves as both BoxVLM.pgraph (prefill) and BoxVLM.graph (branch passes)."""

    def __init__(self, vlms: dict, prefix_model: str, layers: list[int]):
        self.vlms, self.prefix_model, self.layers = vlms, prefix_model, sorted(set(layers))
        self.image_token_id = vlms[prefix_model].config.image_token_id
        self.prefills, self.branches, self.cur, self.pool = {}, {}, None, None

    @staticmethod
    def _masks(text, embeds, mask2d, cache):
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

        kw = dict(config=text.config, inputs_embeds=embeds, attention_mask=mask2d, past_key_values=cache, position_ids=None)
        return {"full_attention": create_causal_mask(**kw), "linear_attention": create_recurrent_attention_mask(**kw)}

    def _prefill(self, pi):
        from transformers import DynamicCache
        from transformers.vision_utils import (get_vision_cu_seqlens, get_vision_interpolation_indices_and_weights,
                                               get_vision_position_ids)

        ids, grid = pi["input_ids"], pi["image_grid_thw"]
        key = (tuple(grid.flatten().tolist()), ids.shape[1])
        if key in self.prefills:
            return self.prefills[key]
        assert ids.shape[0] == 1 and bool(pi["attention_mask"].all()), "batch-1, unpadded prefill only"
        model = self.vlms[self.prefix_model].model
        vis, text = model.visual, model.language_model
        ii, iw = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=vis.num_grid_per_side, mode=vis.interpolation_mode,
            align_corners=vis.interpolation_align_corners, spatial_merge_size=vis.config.spatial_merge_size)
        pos, _ = model.get_rope_index(ids, image_grid_thw=grid, attention_mask=pi["attention_mask"],
                                      mm_token_type_ids=pi["mm_token_type_ids"])
        emb0 = text.embed_tokens.weight.new_zeros(1, ids.shape[1], text.config.hidden_size)
        s = SimpleNamespace(pix=pi["pixel_values"].clone(), grid=grid.clone(), ids=ids.clone(), pos=pos, ii=ii, iw=iw,
                            vpos=get_vision_position_ids(grid, vis.spatial_merge_size), cu=get_vision_cu_seqlens(grid).cpu(),
                            img=(ids[0] == self.image_token_id).nonzero()[:, 0], key=key,
                            masks=self._masks(text, emb0, pi["attention_mask"], DynamicCache(config=text.config)),
                            mask2d=pi["attention_mask"].clone(), start=pos[0, :, -1] + 1)

        def run():
            v = vis(s.pix.type(vis.dtype), grid_thw=s.grid, cu_seqlens=s.cu, position_ids=s.vpos,
                    interp_indices=s.ii, interp_weights=s.iw)
            h = text.embed_tokens(s.ids)
            h = h.index_put((torch.zeros_like(s.img), s.img), v.pooler_output.to(h.dtype))
            out = text(input_ids=None, inputs_embeds=h, position_ids=s.pos, attention_mask=s.masks,
                       past_key_values=DynamicCache(config=text.config), use_cache=True, output_hidden_states=True)
            s.hidden_states, s.cache = out.hidden_states, out.past_key_values

        s.graph = _capture(run, self.pool)
        self.pool = s.graph.pool()
        self.prefills[key] = s
        return s

    @torch.no_grad()
    def __call__(self, pi):
        """BoxVLM.pgraph contract (see GraphedPrefill.__call__)."""
        s = self._prefill(pi)
        s.pix.copy_(pi["pixel_values"])
        s.ids.copy_(pi["input_ids"])
        s.graph.replay()
        return SimpleNamespace(hidden_states=s.hidden_states, past_key_values=s.cache, position_ids=s.pos,
                               image_index=s.img, key=s.key)

    def load_prefix(self, cache, prefix_mask):
        """BoxVLM.graph contract: the latest prefill (always a graph-owned cache here) is the prefix of later branches."""
        self.cur = next(p for p in self.prefills.values() if p.cache is cache)

    @torch.no_grad()
    def branch(self, which, start, branch_ids, row_image):
        """BoxVLM.branch_pass contract for one row over the latest prefill."""
        from locus.models.boxvlm import _fork_cache

        p = self.cur
        R, Lb = branch_ids.shape
        assert R == 1, "GraphedHybrid branch passes are batch-1 (latency path)"
        key = (p.key, which, Lb)
        s = self.branches.get(key)
        if s is None:
            lm = self.vlms[which].model
            rows = torch.zeros(1, dtype=torch.long, device=branch_ids.device)
            mask2d = torch.cat([p.mask2d, torch.ones_like(branch_ids)], 1)
            probe = _fork_cache(p.cache)
            probe.batch_select_indices(rows)
            emb0 = p.hidden_states[0].new_zeros(1, Lb, p.hidden_states[0].shape[-1])
            s = SimpleNamespace(ids=branch_ids.clone(), rows=rows, masks=self._masks(lm.language_model, emb0, mask2d, probe),
                                pos=(p.start[rows][:, None] + torch.arange(Lb, device=rows.device)[None])
                                .to(p.start.dtype)[None].expand(3, -1, -1).contiguous())

            def run():
                c = _fork_cache(p.cache)
                c.batch_select_indices(s.rows)
                out = lm(input_ids=s.ids, past_key_values=c, use_cache=True, attention_mask=s.masks,
                         position_ids=s.pos, output_hidden_states=True)
                s.out = SimpleNamespace(last_hidden_state=out.last_hidden_state,
                                        hidden_states={i: out.hidden_states[i] for i in self.layers})

            s.graph = _capture(run, self.pool)
            self.pool = s.graph.pool()
            self.branches[key] = s
        s.ids.copy_(branch_ids)
        s.graph.replay()
        return s.out


class _HybridBranches:
    """BoxVLM.graph view of a GraphedHybrid (branch-pass call signature)."""

    def __init__(self, g: GraphedHybrid):
        self.g = g

    def load_prefix(self, cache, prefix_mask):
        self.g.load_prefix(cache, prefix_mask)

    def __call__(self, which, start, branch_ids, row_image):
        return self.g.branch(which, start, branch_ids, row_image)
