"""VLM + [BOX] query + cross-attention box decoder.

Two input layouts:
  forward(inputs)          one question per row: [image][instruction][BOX]
  forward_shared(...)      shared prefill: each image + template prefix is run once; its questions are
                           isolated branches over the cached prefix (each branch sees the prefix and
                           itself only, all branches start at the same position). Qwen3-VL only: its
                           KV cache is concatenation-based and differentiable; Qwen3.5's linear-attention
                           cache updates state tensors in place.

The [BOX] query is the backbone's `<|box_start|>` token plus a learned embedding offset, so no vocab
resize is needed; with n_query > 1 the token is repeated, each copy with its own offset ("pause"/register
tokens: more in-LLM compute for the box). The decoder reads the [BOX] states and the image-token states
at a mid-depth layer and the deepest layer run. exit_layer truncates the LLM (early exit): layers past
the ones the decoder reads are never computed. LoRA (S1) goes on every linear layer of the language
model; the vision tower stays frozen.

alora=True (shared layout only) is Activated LoRA (Greenewald et al. 2025): the prefix runs through the
base model with the adapter disabled and only the box branch tokens pass through the adapter. The prefix
cache, and every decision branch over it, is then exactly the base model's, so decision answers are
untouched by construction (a box-only LoRA otherwise degrades them; see results/box_train/cmp_*.json).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import cache_utils

from locus.models.backbone.adapters import VLMAdapter
from locus.models.graphed import GraphedBranches, GraphedHybrid, GraphedPrefill, _HybridBranches
from locus.models.heads.box_decoder import BoxDecoder, MLPHead


def _linear_batch_select(self, indices: torch.Tensor) -> None:
    """Row selection for linear-attention (Qwen3.5 Gated DeltaNet) cache layers, which transformers leaves out:
    conv and recurrent states are per-row, like KV. Hybrid layers also select their keys/values."""
    for d in (self.conv_states, self.recurrent_states):
        for i, t in d.items():
            if t is not None:
                d[i] = t[indices, ...]
    if getattr(self, "keys", None) is not None and self.keys.numel():
        self.keys, self.values = self.keys[indices, ...], self.values[indices, ...]


def _linear_update_recurrent(self, recurrent_states: torch.Tensor, state_idx: int = 0, **kwargs) -> torch.Tensor:
    """Out-of-place recurrent-state update. transformers copies in place (static address for CUDA graphs), which
    overwrites the tensor the delta-rule kernel saved as initial_state when training branches over a prefix."""
    self.recurrent_states[state_idx] = recurrent_states
    self.is_recurrent_states_initialized[state_idx] = True
    return recurrent_states


def _fork_cache(cache):
    """A copy of the cache structure sharing its tensors. Branch passes only rebind per-layer tensors (indexing,
    concatenation, the out-of-place recurrent update above), so the prefix stays intact for reuse. Unlike deepcopy,
    this also works on a prefix that carries gradients (training with the adapter on the prefix)."""
    new = copy.copy(cache)
    new.layers = []
    for layer in cache.layers:
        lc = copy.copy(layer)
        for k in ("conv_states", "recurrent_states", "has_previous_state"):
            if isinstance(getattr(layer, k, None), dict):
                setattr(lc, k, dict(getattr(layer, k)))
        new.layers.append(lc)
    return new


cache_utils.LinearAttentionLayer.batch_select_indices = _linear_batch_select
cache_utils.LinearAttentionLayer.update_recurrent_state = _linear_update_recurrent

BOX_TOKEN = "<|box_start|>"
LORA_TARGETS = ("q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
                "|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj")  # last five: Qwen3.5 linear-attention layers
_SPLIT = "⁣EXPR⁣"  # sentinel used to cut the prompt template into prefix / branch


class BoxVLM(nn.Module):
    def __init__(self, backbone: str, device: str = "cuda", max_pixels: int = 640 * 640, lora_r: int = 0,
                 mid: float = 0.5, decoder_layers: int = 3, grad_ckpt: bool = False, exit_layer: float = 1.0,
                 n_query: int = 1, refine: bool = False, head: str = "xattn", dfl: int = 0,
                 alora: bool = False, merger_lora: bool = False, full_ft: bool = False):
        super().__init__()
        self.ad = VLMAdapter(backbone, device, max_pixels)
        model = self.ad.model
        for p in model.parameters():
            p.requires_grad_(False)
        lm = model.model.language_model
        n_layers = model.config.get_text_config().num_hidden_layers
        n_run = round(exit_layer * n_layers)
        if n_run < n_layers:
            lm.layers = lm.layers[:n_run]
            lm.config.num_hidden_layers = n_run
        self.n_layers_run, self.n_layers_total = n_run, n_layers
        self.layers = [round(mid * n_run), n_run]  # hidden_states index: 0 = embeddings
        if grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if lora_r:
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                             target_modules=rf".*language_model.*\.({LORA_TARGETS})"
                             + (r"|.*visual\..*merger.*\.linear_fc[12]" if merger_lora else ""))
            model = get_peft_model(model, cfg)
            self.ad.model = model
        self.lora_r = lora_r
        self.alora = alora and bool(lora_r)
        assert not (merger_lora and self.alora), "a merger adapter changes the prefix; aLoRA needs it to stay base"
        # Full fine-tuning (Jag's recipe) of the language model and vision merger(s): dual mode only, the base copy
        # answers decisions. Same inference memory as enable_fast(), which keeps base + merged copies anyway.
        assert not (full_ft and lora_r), "--full-ft replaces the adapter"
        self.full_ft = full_ft
        if full_ft:
            for n, p in model.named_parameters():
                if ".language_model." in n or ".merger" in n:
                    p.requires_grad_(True)
        self.fast: dict | None = None  # enable_fast(): plain (wrapper-free) models for inference
        self.graph: GraphedBranches | None = None  # enable_fast(graphs=True)
        self.pgraph: GraphedPrefill | None = None  # enable_fast(graphs=True, graph_prefill=True)
        self._xy_cache: dict = {}
        self.base = model.get_base_model() if lora_r else model
        self.box_id = self.ad.tok.convert_tokens_to_ids(BOX_TOKEN)
        self.n_query = n_query
        d = model.config.get_text_config().hidden_size
        self.box_embed = nn.Parameter(torch.zeros(n_query, d, device=device))
        self.base.model.language_model.embed_tokens.register_forward_hook(self._add_box_embed)
        head_cls = {"xattn": BoxDecoder, "mlp": MLPHead}[head]
        self.decoder = head_cls(d, n_feat=len(self.layers), layers=decoder_layers, n_query=n_query,
                                refine=refine, **({"dfl": dfl} if dfl else {})).to(device)
        pre, post = self.prompt(_SPLIT).split(_SPLIT)
        self.prefix, self.suffix = pre.rstrip(" "), post  # branch text starts with " <expression>"

    def _add_box_embed(self, module, args, out):
        is_box = args[0] == self.box_id
        k = (is_box.cumsum(1) - 1).clamp(0, self.n_query - 1)  # which copy of the query token
        return out + is_box[..., None] * self.box_embed[k].to(out.dtype)

    def prompt(self, text: str) -> str:
        return self.ad.template(f"Locate: {text}", generation_prompt=True) + BOX_TOKEN * self.n_query

    def trainable(self) -> dict[str, list[nn.Parameter]]:
        return dict(head=[self.box_embed, *self.decoder.parameters()],
                    lora=[p for n, p in self.ad.model.named_parameters() if "lora_" in n or (self.full_ft and p.requires_grad)])

    def _xy(self, thw, c, device):
        m = self.ad.model.config.vision_config.spatial_merge_size
        gh, gw = thw[1] // m, thw[2] // m
        assert gh * gw == c, (gh, gw, c)
        ys, xs = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
        return torch.stack([(xs.flatten() + 0.5) / gw, (ys.flatten() + 0.5) / gh], -1)

    def _vis(self, hs, ids, grid_thw):
        """Per-row image-token states (padded) + centres + mask from a pass that contains the images."""
        mask = ids == self.ad.image_token_id
        counts = mask.sum(1).tolist()
        n, B = max(counts), ids.shape[0]
        vis = [h.new_zeros(B, n, h.shape[-1]) for h in hs]
        vmask = torch.zeros(B, n, dtype=torch.bool, device=ids.device)
        xy = torch.zeros(B, n, 2, device=ids.device)
        for i, (c, thw) in enumerate(zip(counts, grid_thw.tolist())):
            for v, h in zip(vis, hs):
                v[i, :c] = h[i, mask[i]]
            vmask[i, :c] = True
            xy[i, :c] = self._xy(thw, c, ids.device)
        return vis, xy, vmask

    def forward(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """inputs: processor output for rows [image][prompt(text)], left-padded → (xyxy [B,4], conf logits [B])."""
        out = self.base.model(**inputs, output_hidden_states=True, use_cache=False)
        hs = [out.hidden_states[i] for i in self.layers]
        ids = inputs["input_ids"]
        assert (ids[:, -1] == self.box_id).all()
        vis, xy, vmask = self._vis(hs, ids, inputs["image_grid_thw"])
        return self.decoder([x[:, -self.n_query:] for x in hs], vis, xy, vmask)

    def enable_fast(self, graphs: bool = False, max_prefix: int = 512, graph_prefill: bool = False) -> None:
        """Inference without PEFT wrappers: a plain base model (prefix, base-model verifier) and a plain copy with the
        LoRA merged into its weights (box branches). The PEFT path launches ~5 kernels per adapted linear, keeps the
        adapter in fp32 (re-cast every call) and runs wrapper Python even with the adapter disabled.
        graphs=True also replays branch passes as CUDA graphs (graphed.py) over the latest prefill's prefix.
        graph_prefill=True (with graphs) also replays batch-1 prefills as CUDA graphs (GraphedPrefill)."""
        if self.lora_r:
            merged = copy.deepcopy(self.ad.model).merge_and_unload().to(torch.bfloat16).eval()
            merged.model.language_model.embed_tokens.register_forward_hook(self._add_box_embed)
            base = copy.deepcopy(self.ad.model).unload().to(torch.bfloat16).eval() if self.alora else merged
        elif graph_prefill:  # no adapter (full FT / frozen): one plain model serves both roles
            base = merged = self.base
        else:
            return
        self.fast = dict(base=base, adapted=merged)
        if graphs and "linear_attention" in getattr(self.base.config.get_text_config(), "layer_types", []):
            assert graph_prefill, "hybrid backbones graph branches over a graphed prefill only (GraphedHybrid)"
            self.pgraph = GraphedHybrid({"base": base, "adapted": merged}, "base" if self.alora else "adapted", self.layers)
            self.graph = _HybridBranches(self.pgraph)
            return
        if graphs:
            self.graph = GraphedBranches({"base": base, "adapted": merged}, self.layers, max_prefix)
            if graph_prefill:
                self.pgraph = GraphedPrefill(self._lm(adapter=not self.alora), self.layers)

    def _lm(self, adapter: bool):
        """Inner language model (+ vision) to run: adapted or base weights, plain modules when enable_fast() was called."""
        if self.fast is not None:
            return self.fast["adapted" if adapter else "base"]
        return self.base

    def prefill(self, prefix_inputs) -> dict:
        """Run [image][prefix] once per image (left-padded; base model when alora). Returns the state branches build on."""
        pi = prefix_inputs
        if self.pgraph is not None and pi["input_ids"].shape[0] == 1:
            out = self.pgraph(pi)
            self.graph.load_prefix(out.past_key_values, pi["attention_mask"])
            img = out.image_index
            key = out.key
            if key not in self._xy_cache:
                self._xy_cache[key] = self._xy(key[0], img.shape[0], img.device)[None]
            return dict(cache=out.past_key_values, vis=[out.hidden_states[i][:, img] for i in self.layers],
                        xy=self._xy_cache[key], vmask=torch.ones(1, img.shape[0], dtype=torch.bool, device=img.device),
                        start=out.position_ids[0, :, -1] + 1, mask=pi["attention_mask"])
        if self.fast is not None:
            with torch.no_grad():
                out = self._lm(adapter=not self.alora).model(**pi, output_hidden_states=True, use_cache=True)
            lm = self.base.model
            if self.graph is not None:
                self.graph.load_prefix(out.past_key_values, pi["attention_mask"])
        elif self.alora:
            lm = self.base.model  # base model, nothing trainable upstream → no graph needed
            with self.ad.model.disable_adapter(), torch.no_grad():
                out = lm(**pi, output_hidden_states=True, use_cache=True)
        else:
            lm = self.base.model
            out = lm(**pi, output_hidden_states=True, use_cache=True)
        hs = [out.hidden_states[i] for i in self.layers]
        vis, xy, vmask = self._vis(hs, pi["input_ids"], pi["image_grid_thw"])
        pos, _ = lm.get_rope_index(pi["input_ids"], image_grid_thw=pi["image_grid_thw"],
                                   attention_mask=pi["attention_mask"], mm_token_type_ids=pi["mm_token_type_ids"])
        start = pos[0, :, -1] + 1  # left padding → last column is each prefix's last real token
        return dict(cache=out.past_key_values, vis=vis, xy=xy, vmask=vmask, start=start, mask=pi["attention_mask"])

    def branch_pass(self, state: dict, branch_ids, branch_mask, row_image, keep_cache: bool = False, adapter: bool = True):
        """One pass of right-padded branches over the cached prefix; each branch sees the prefix and itself only.
        keep_cache=True works on a copy so the prefix cache can be reused (inference). adapter=False (fast mode only)
        runs the plain base model; on the PEFT path the caller uses disable_adapter()."""
        if self.graph is not None:
            return self.graph("adapted" if adapter else "base", state["start"], branch_ids, row_image)
        lm = self._lm(adapter).model if self.fast is not None else self.base.model
        cache = _fork_cache(state["cache"]) if keep_cache else state["cache"]
        cache.batch_select_indices(row_image)
        Lb = branch_ids.shape[1]
        text_pos = (state["start"][row_image][:, None] + torch.arange(Lb, device=row_image.device)[None]).to(state["start"].dtype)
        kw = dict(input_ids=branch_ids, past_key_values=cache, use_cache=True,
                  attention_mask=torch.cat([state["mask"][row_image], branch_mask], 1),
                  position_ids=text_pos[None].expand(3, -1, -1))
        return lm(**kw, output_hidden_states=True)

    def forward_shared(self, prefix_inputs, branch_ids, branch_mask, row_image, state: dict | None = None):
        """prefix_inputs: processor output for [image][prefix] per image (left-padded);
        branch_ids/branch_mask [R, Lb]: tokenised " <expr>" + suffix per question (right-padded);
        row_image [R]: index of each question's image. → (xyxy [R,4], conf logits [R]).
        Pass a prefill() state to reuse a cached prefix (its cache is then left untouched)."""
        keep = state is not None
        state = state or self.prefill(prefix_inputs)
        b = self.branch_pass(state, branch_ids, branch_mask, row_image, keep_cache=keep)
        last = branch_mask.sum(1) - 1  # right padding → last real token = last [BOX] copy
        idx = last[:, None] - torch.arange(self.n_query - 1, -1, -1, device=last.device)[None]  # [R, n_query]
        r = torch.arange(len(last), device=last.device)[:, None]
        hq = [b.hidden_states[i][r, idx] for i in self.layers]
        return self.decoder(hq, [v[row_image] for v in state["vis"]], state["xy"][row_image], state["vmask"][row_image])

    def tokenize_branches(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = [self.ad.tok.encode(" " + t + self.suffix, add_special_tokens=False) for t in texts]
        L = max(map(len, enc))
        ids = torch.full((len(enc), L), self.ad.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros(len(enc), L, dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, : len(e)], mask[i, : len(e)] = torch.tensor(e), 1
        return ids, mask

    def state_dict_trainable(self) -> dict:
        sd = {"box_embed": self.box_embed.detach().cpu(), "decoder": self.decoder.state_dict()}
        if self.lora_r or self.full_ft:  # full_ft: the fine-tuned weights, under the same key (loaded by name)
            sd["lora"] = {n: p.detach().to(torch.bfloat16).cpu() if self.full_ft else p.detach().cpu()
                          for n, p in self.ad.model.named_parameters() if "lora_" in n or (self.full_ft and p.requires_grad)}
        return sd
