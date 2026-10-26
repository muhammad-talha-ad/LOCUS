"""Box head as a speculative drafter for coordinate tokens (Jacobi verification over a shared prefix).

The box head answers every question of an image in one branch pass over the cached image prefix. Its boxes are
rendered as the backbone's own coordinate-token answer ("x0, y0, x1, y1]" in 0–1000 units, after a teacher-forced
JSON preamble) and verified in parallel by the backbone's LM head, with the adapter disabled (base model):

  pass k:  argmax g_t of every draft position given the draft before it, all questions batched as branches.
           Accept the longest prefix with g_t == draft_t; the token after it, g_n, is exact too.
           Next draft = accepted + g_n + g_{n+1..} (Jacobi iterate). Stop when nothing changes.

The fixed point is exactly the greedy decode under the same computation. Rows still unconverged after max_passes
finish with plain greedy decoding from their accepted prefix, so the output is always the greedy answer; the cost
is the number of sequential passes. max_passes=0 is plain greedy decoding (the coordinate-token control on the same
prompt, cache and kernels).

accept < 1 relaxes the check (typical acceptance, as in Medusa): a draft token also passes when the verifier gives it
at least accept x the probability of its argmax. The output is then no longer the greedy answer but the head's box
unless the verifier clearly disagrees; accept=1 is the exact rule above.
"""
from __future__ import annotations

import re

import numpy as np
import torch

from locus.models.boxvlm import BOX_TOKEN, BoxVLM

# Continues BoxVLM.prefix ("...Locate:"): the zero-shot grounding prompt of the Qwen harness, with a colon.
VERIFY_PROMPT = ' "{}" in the image and output its bbox coordinates in JSON format.'
PREAMBLE = '```json\n[\n\t{"bbox_2d": ['


class DraftVerify:
    def __init__(self, model: BoxVLM, max_passes: int = 4, max_tokens: int = 24, verifier_adapter: bool = False,
                 accept: float = 1.0):
        assert model.n_layers_run == model.n_layers_total, "the verifier needs the full LM (no early exit)"
        self.m, self.tok = model, model.ad.tok
        self.max_passes, self.max_tokens = max_passes, max_tokens
        self.close = self.tok.encode("]", add_special_tokens=False)[0]
        self.vsuffix = model.suffix.replace(BOX_TOKEN, "") + PREAMBLE
        self.verifier_adapter = verifier_adapter  # aLoRA trained with --ar-weight: adapter on verifier branches too
        self.log_accept = float(np.log(accept)) if accept < 1 else None
        self._closes = {}

    # -- token helpers ------------------------------------------------------
    def is_close(self, t: int) -> bool:
        if t not in self._closes:
            self._closes[t] = self.tok.decode([t]).lstrip().startswith("]")
        return self._closes[t]

    def cut(self, ids: list[int]) -> list[int]:
        """Truncate after the first token that closes the coordinate list; append "]" if there is none."""
        for i, t in enumerate(ids):
            if self.is_close(t):
                return ids[: i + 1]
        return ids + [self.close]

    def render(self, box) -> list[int]:
        return self.tok.encode(", ".join(str(int(round(min(max(float(v), 0.0), 1.0) * 1000))) for v in box) + "]",
                               add_special_tokens=False)

    def parse(self, ids: list[int]) -> np.ndarray:
        m = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", self.tok.decode(ids))
        if m is None:
            return np.full(4, np.nan)
        x0, y0, x1, y1 = (min(int(v), 1000) / 1000 for v in m.groups())
        return np.array([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])

    def verifier_inputs(self, texts, boxes=None):
        """Verifier-branch prompt tokens per question (continuing BoxVLM.prefix), and rendered answers if boxes given."""
        prompts = [self.tok.encode(VERIFY_PROMPT.format(t) + self.vsuffix, add_special_tokens=False) for t in texts]
        return prompts, None if boxes is None else [self.render(b) for b in boxes]

    # -- verifier passes ----------------------------------------------------
    def _greedy_at(self, state, prompts, drafts, rows, row_image):
        """One batched branch pass: argmax at every draft position (and the one after the last) for the given rows.
        With accept < 1, a draft token the verifier finds plausible enough is returned in place of the argmax."""
        seqs = [prompts[i] + drafts[i] for i in rows]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros(len(seqs), L, dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, : len(s)], mask[j, : len(s)] = torch.tensor(s), 1
        dev = row_image.device
        if self.m.fast is not None:
            h = self.m.branch_pass(state, ids.to(dev), mask.to(dev), row_image[rows], keep_cache=True,
                                   adapter=self.verifier_adapter).last_hidden_state
        else:
            ctx = self.m.ad.model.disable_adapter() if self.m.lora_r and not self.verifier_adapter else torch.no_grad()
            with ctx:
                h = self.m.branch_pass(state, ids.to(dev), mask.to(dev), row_image[rows], keep_cache=True).last_hidden_state
        spans = [(j, len(prompts[i]) - 1, len(prompts[i]) + len(drafts[i])) for j, i in enumerate(rows)]
        sel = torch.cat([h[j, a:b] for j, a, b in spans])  # LM head only where needed: len(draft) + 1 per row
        logits = self.m.base.lm_head(sel).float()
        pred = logits.argmax(-1)
        if self.log_accept is not None:
            draft = torch.tensor([t for i in rows for t in drafts[i] + [-1]], device=pred.device)
            lp = logits.log_softmax(-1)
            ok = (draft >= 0) & (lp.gather(1, draft.clamp(min=0)[:, None])[:, 0] - lp.max(-1).values >= self.log_accept)
            pred = torch.where(ok, draft, pred)
        pred = pred.tolist()
        out, k = [], 0
        for _, a, b in spans:
            out.append(pred[k : k + b - a])
            k += b - a
        return out

    @torch.no_grad()
    def __call__(self, prefix_inputs, texts: list[str], row_image: torch.Tensor, verify: bool = True,
                 draft: np.ndarray | None = None) -> dict:
        """`draft` (boxes from another model's head) replaces this model's head as the drafter."""
        m = self.m
        state = m.prefill(prefix_inputs)
        bid, bmask = m.tokenize_branches(texts)
        dev = row_image.device
        head, conf = m.forward_shared(None, bid.to(dev), bmask.to(dev), row_image, state=state)
        head = head.float().cpu().numpy()
        res = dict(head=head, conf=torch.sigmoid(conf.float()).cpu().numpy())
        if not verify:
            return res
        assert m.alora or not m.lora_r or self.verifier_adapter, \
            "the base-model verifier needs a base prefix: use an aLoRA (or frozen) head, or a jointly trained adapter"
        prompts = self.verifier_inputs(texts)[0]
        R = len(texts)
        drafts = [self.render(b) for b in (head if draft is None else draft)] if self.max_passes else [[] for _ in range(R)]
        acc = [0] * R  # length of the greedy-exact prefix of each draft
        passes = np.zeros(R, int)
        active = list(range(R)) if self.max_passes else []
        for _ in range(self.max_passes):
            if not active:
                break
            g = self._greedy_at(state, prompts, drafts, active, row_image)
            nxt = []
            for i, gi in zip(active, g):
                passes[i] += 1
                d = drafts[i]
                n = next((k for k, (a, b) in enumerate(zip(d, gi)) if a != b and not (self.is_close(a) and self.is_close(b))), len(d))
                if n == len(d):  # fixed point: every draft token is the greedy token
                    acc[i] = len(d)
                    continue
                acc[i] = n + 1  # g_n is exact as well
                drafts[i] = self.cut(d[:n] + gi[n:])
                nxt.append(i)
            active = nxt
        # Unconverged rows (and max_passes=0): plain greedy decoding from the exact prefix.
        todo = [i for i in range(R) if acc[i] < len(drafts[i]) or not drafts[i]]
        for i in todo:
            drafts[i] = drafts[i][: acc[i]]
        while todo:
            g = self._greedy_at(state, prompts, [d for d in drafts], todo, row_image)
            nxt = []
            for i, gi in zip(todo, g):
                passes[i] += 1
                drafts[i] = drafts[i] + [gi[-1]]
                if not self.is_close(gi[-1]) and len(drafts[i]) < self.max_tokens:
                    nxt.append(i)
            todo = nxt
        res.update(box=np.stack([self.parse(d) for d in drafts]), passes=passes, tokens=np.array([len(d) for d in drafts]))
        return res
