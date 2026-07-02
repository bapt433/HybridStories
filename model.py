import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    d_model: int = 32
    n_layers: int = 17
    vocab_size: int = 2048
    n_heads_q: int = 1
    n_heads_kv: int = 1
    d_state: int = 4
    ssm_expand: int = 2
    mlp_ratio: float = 2.77
    rope_theta: float = 500_000.0
    max_seq_len: int = 128
    d_conv: int = 4
    use_checkpoint: bool = True
    n_meta_tokens: int = 2
    block_sharing: bool = True

    @classmethod
    def from_dict(cls, d):
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @property
    def d_head(self):
        return self.d_model // self.n_heads_q

    @property
    def d_inner(self):
        return self.ssm_expand * self.d_model

    @property
    def mlp_hidden(self):
        return int(self.d_model * self.mlp_ratio)

    @property
    def dt_rank(self):
        return math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


def precompute_rope(dim, max_seq_len, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(q, k, cos, sin):
    d_head = q.shape[-1]
    half = d_head // 2
    cos = cos[: q.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[: q.shape[2]].unsqueeze(0).unsqueeze(0)
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_out = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_out = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_out, k_out


class GQA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads_q = config.n_heads_q
        self.n_heads_kv = config.n_heads_kv
        self.d_head = config.d_head
        self.n_rep = config.n_heads_q // config.n_heads_kv

        self.Wq = nn.Linear(config.d_model, config.n_heads_q * self.d_head, bias=False)
        self.Wk = nn.Linear(config.d_model, config.n_heads_kv * self.d_head, bias=False)
        self.Wv = nn.Linear(config.d_model, config.n_heads_kv * self.d_head, bias=False)
        self.Wo = nn.Linear(config.n_heads_q * self.d_head, config.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.Wq(x).view(B, T, self.n_heads_q, self.d_head).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_heads_kv, self.d_head).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads_kv, self.d_head).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.n_heads_kv, self.n_rep, T, self.d_head)
            k = k.reshape(B, self.n_heads_q, T, self.d_head)
            v = v.unsqueeze(2).expand(B, self.n_heads_kv, self.n_rep, T, self.d_head)
            v = v.reshape(B, self.n_heads_q, T, self.d_head)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.Wo(out)


class MambaSSM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.d_inner = config.d_inner
        self.dt_rank = config.dt_rank

        self.in_proj = nn.Linear(config.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=config.d_conv,
            bias=True, padding=config.d_conv - 1, groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, 2 * config.d_state + self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32)
        A = A.view(1, config.d_state).expand(self.d_inner, config.d_state).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_proj_out = self.x_proj(x_conv)
        B_p, C_p, dt = x_proj_out.split(
            [self.d_state, self.d_state, self.dt_rank], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)

        y = self._selective_scan(x_conv, dt, A, B_p, C_p)

        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        return self.out_proj(y)

    def _selective_scan(self, x, dt, A, B, C):
        in_dtype = x.dtype
        x32 = x.float()
        dt32 = dt.float()
        B32 = B.float()
        C32 = C.float()

        Bsz, L, d_inner = x32.shape
        d_state = A.shape[1]

        dA = torch.exp(dt32.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt32.unsqueeze(-1) * B32.unsqueeze(2) * x32.unsqueeze(-1)

        h = torch.zeros(Bsz, d_inner, d_state, device=x32.device, dtype=torch.float32)
        y = torch.empty(Bsz, L, d_inner, device=x32.device, dtype=torch.float32)

        for t in range(L):
            h = dA[:, t] * h + dB[:, t]
            y[:, t] = (h * C32[:, t].unsqueeze(1)).sum(dim=-1)

        return y.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class HybridBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = GQA(config)
        self.ssm = MambaSSM(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, config.mlp_hidden)

    def forward(self, x, cos, sin):
        normed = self.norm1(x)
        x = x + self.attn(normed, cos, sin) + self.ssm(normed)
        x = x + self.mlp(self.norm2(x))
        return x


class HybridLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        if config.n_meta_tokens > 0:
            self.meta_tokens = nn.Parameter(
                torch.randn(config.n_meta_tokens, config.d_model) * 0.02
            )
        else:
            self.meta_tokens = None

        rope_len = config.max_seq_len + config.n_meta_tokens
        cos, sin = precompute_rope(config.d_head, rope_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=True)
        self.register_buffer("rope_sin", sin, persistent=True)

        if config.block_sharing and config.n_layers >= 4:
            n_unique = (config.n_layers + 1) // 2
            unique_blocks = nn.ModuleList([HybridBlock(config) for _ in range(n_unique)])
            self.blocks = nn.ModuleList()
            for i in range(config.n_layers):
                self.blocks.append(unique_blocks[i % n_unique])
        else:
            self.blocks = nn.ModuleList([HybridBlock(config) for _ in range(config.n_layers)])

        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)

        if self.meta_tokens is not None:
            meta = self.meta_tokens.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([meta, x], dim=1)

        cos = self.rope_cos[:x.shape[1]]
        sin = self.rope_sin[:x.shape[1]]

        for block in self.blocks:
            if self.config.use_checkpoint and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)

        x = self.norm_f(x)

        if self.meta_tokens is not None:
            x = x[:, self.config.n_meta_tokens:, :]

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.9):
        self.eval()
        for _ in range(max_new_tokens):
            T = idx.shape[1]
            if T > self.config.max_seq_len:
                idx_cond = idx[:, -self.config.max_seq_len:]
            else:
                idx_cond = idx

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                indices_to_remove = mask.scatter(-1, sorted_indices, mask)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

        self.train()
        return idx


def count_parameters(model):
    seen = set()
    total = 0
    for name, param in model.named_parameters():
        if id(param) not in seen:
            seen.add(id(param))
            total += param.numel()
    return total


def param_breakdown(model):
    components = {}
    seen_ids = {}

    for name, param in model.named_parameters():
        parts = name.split(".")
        if parts[0] == "tok_emb":
            comp = "embedding (tied)"
        elif parts[0] == "lm_head":
            comp = "lm_head (tied, shared)"
        elif parts[0] == "norm_f":
            comp = "final_norm"
        elif parts[0] == "blocks":
            layer_idx = parts[1]
            sub = parts[2] if len(parts) > 2 else "other"
            comp = f"block_{layer_idx}/{sub}"
        else:
            comp = parts[0]

        pid = id(param)
        if pid not in seen_ids:
            seen_ids[pid] = True
            components[comp] = components.get(comp, 0) + param.numel()

    return components


if __name__ == "__main__":
    config = ModelConfig()
    model = HybridLM(config)
    total = count_parameters(model)

    print(f"d_model     = {config.d_model}")
    print(f"n_layers    = {config.n_layers}")
    print(f"vocab_size  = {config.vocab_size}")
    print(f"d_head      = {config.d_head}")
    print(f"d_inner     = {config.d_inner}")
    print(f"d_state     = {config.d_state}")
    print(f"mlp_hidden  = {config.mlp_hidden}")
    print(f"dt_rank     = {config.dt_rank}")
    print(f"n_heads_q   = {config.n_heads_q}")
    print(f"n_heads_kv  = {config.n_heads_kv}")
    print(f"\nTotal unique parameters: {total:,} ({total / 1e6:.2f}M)")

    breakdown = param_breakdown(model)
    per_block_keys = [k for k in breakdown if k.startswith("block_0/")]
    if per_block_keys:
        layer_total = sum(breakdown[k] for k in breakdown if k.startswith("block_0/"))
        print(f"Per-layer params:     {layer_total:,}")
        for k in sorted(per_block_keys):
            print(f"  {k:30s} {breakdown[k]:>10,}")
    print(f"Embedding (tied):     {breakdown.get('embedding (tied)', 0):,}")
    print(f"Final norm:           {breakdown.get('final_norm', 0):,}")