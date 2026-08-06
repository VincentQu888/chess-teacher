"""Attention-based policy/value network.

The board is 64 square tokens + 1 global-feature token + 1 CLS token. A stack of
transformer encoder blocks self-attends over these tokens. Two things come out:

* policy logits (per square x 73 AlphaZero move planes) and a scalar value, used
  for MCTS and play;
* the per-layer attention weights, exposed as the *attention-weighted board
  state*. Following HEX-RL (arXiv:2112.08907), the attention that the CLS/value
  token and the chosen move's from/to squares place on other squares tells us
  which board elements most influenced the decision -- the raw material for an
  inherently-explainable move rationale.

The attention module is hand-written (rather than nn.MultiheadAttention) so we
can return the attention probabilities cheaply for the explainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import (
    NUM_GLOBAL_FEATURES,
    NUM_PIECE_IDS,
    NUM_SQUARES,
    POLICY_SIZE,
)

GLOBAL_TOKEN = NUM_SQUARES        # index 64
CLS_TOKEN = NUM_SQUARES + 1       # index 65
NUM_TOKENS = NUM_SQUARES + 2      # 66


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    wdl: bool = False  # if True, value head predicts win/draw/loss (3 logits)


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if need_weights:
            # explicit path so we can return attention probs for the explainer
            scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
            attn = scores.softmax(dim=-1)
            out = self.drop(attn) @ v
            weights = attn.mean(dim=1)  # avg over heads
        else:
            # fused, MPS/CUDA-accelerated path for training/play (no weights).
            # (MPS SDPA has no attention-dropout support; other dropouts remain.)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            weights = None
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        return out, weights


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = SelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        a, w = self.attn(self.ln1(x), need_weights=need_weights)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, w


class AttentionChessNet(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        d = self.cfg.d_model

        self.piece_emb = nn.Embedding(NUM_PIECE_IDS, d)
        self.square_pos = nn.Embedding(NUM_SQUARES, d)
        self.global_proj = nn.Linear(NUM_GLOBAL_FEATURES, d)
        self.global_type = nn.Parameter(torch.zeros(1, 1, d))
        self.cls_type = nn.Parameter(torch.zeros(1, 1, d))
        self.in_drop = nn.Dropout(self.cfg.dropout)

        self.blocks = nn.ModuleList([EncoderBlock(self.cfg) for _ in range(self.cfg.n_layers)])
        self.ln_f = nn.LayerNorm(d)

        # policy: per square token -> 73 planes -> flatten to 4672
        self.policy_head = nn.Linear(d, 73)
        # value: CLS token -> scalar in (-1, 1), OR win/draw/loss logits (WDL head).
        # WDL (Lc0-style) gives a better-calibrated, non-saturating value signal that
        # directly targets the diagnosed 'value blindness' (over-optimistic evals).
        self.wdl = bool(self.cfg.wdl)
        if self.wdl:
            self.value_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3)
            )
        else:
            self.value_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
            )

        self.register_buffer("_sq_idx", torch.arange(NUM_SQUARES), persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _embed(self, piece_ids: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        B = piece_ids.shape[0]
        sq = self.piece_emb(piece_ids) + self.square_pos(self._sq_idx).unsqueeze(0)  # (B,64,d)
        g = (self.global_proj(globals_).unsqueeze(1) + self.global_type).expand(B, 1, -1)  # (B,1,d)
        cls = self.cls_type.expand(B, 1, -1)  # (B,1,d)
        x = torch.cat([sq, g, cls], dim=1)  # (B, 66, d)
        return self.in_drop(x)

    def forward(
        self,
        piece_ids: torch.Tensor,
        globals_: torch.Tensor,
        return_attn: bool = False,
        return_wdl: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        x = self._embed(piece_ids, globals_)
        attns: List[torch.Tensor] = []
        for blk in self.blocks:
            x, w = blk(x, need_weights=return_attn)
            if return_attn and w is not None:
                attns.append(w)
        x = self.ln_f(x)

        sq_tokens = x[:, :NUM_SQUARES, :]              # (B,64,d)
        policy = self.policy_head(sq_tokens)           # (B,64,73)
        policy = policy.reshape(x.shape[0], POLICY_SIZE)  # index = sq*73 + plane
        vhead = self.value_head(x[:, CLS_TOKEN, :])       # (B,1) or (B,3)
        if self.wdl:
            wdl_logits = vhead                            # (B,3): win/draw/loss
            probs = torch.softmax(wdl_logits, dim=-1)
            value = probs[:, 0] - probs[:, 2]             # scalar in (-1,1), MCTS-compatible
        else:
            wdl_logits = None
            value = torch.tanh(vhead).squeeze(-1)         # (B,)
        if return_wdl:
            return policy, value, (attns if return_attn else None), wdl_logits
        return policy, value, (attns if return_attn else None)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":  # tiny smoke test
    from encoding import encode_position
    import chess

    net = AttentionChessNet()
    print("params:", count_params(net))
    b = chess.Board()
    pids, g, _ = encode_position(b)
    pids_t = torch.from_numpy(pids).unsqueeze(0)
    g_t = torch.from_numpy(g).unsqueeze(0)
    pol, val, attn = net(pids_t, g_t, return_attn=True)
    print("policy", pol.shape, "value", val.shape, "value=", float(val))
    print("attn layers:", len(attn), "shape:", attn[0].shape)
