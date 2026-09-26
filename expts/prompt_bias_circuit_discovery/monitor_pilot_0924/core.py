"""Full-trace likelihood with explicit Qwen3 attention and fixed sparse cell pools.

Isolated from the existing answer-objective runners. Coordinates are reader, source.
"""
from __future__ import annotations
import math
import types
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


class CellPool:
    def __init__(self, record, condition, device='cpu', with_probe=False):
        assert condition in ('rr_on', 'rr_off', 'pr_on', 'pr_off', 'joint')
        self.condition = condition
        self.length = len(record['input_ids'])
        self.prompt_length = record['prompt_length']
        self.chunks = record['chunks']
        size = self.length + (6 if with_probe else 0)
        t = torch.arange(size, device=device)
        q, k = t[:, None], t[None, :]
        self.causal = k <= q
        mutable = (k < q) & (k >= 3) & (q < self.length)
        if condition.startswith('rr'):
            eligible = mutable & (q >= self.prompt_length) & (k >= self.prompt_length)
        elif condition.startswith('pr'):
            eligible = mutable & (q >= self.prompt_length) & (k < self.prompt_length)
        else:
            eligible = mutable
        token_chunk = torch.full((size,), -1, device=device, dtype=torch.long)
        for i, chunk in enumerate(self.chunks):
            token_chunk[chunk['start']:chunk['end'] + 1] = i
        pairs = token_chunk[:, None] * len(self.chunks) + token_chunk[None, :]
        codes = pairs[eligible].unique(sorted=True)
        self.pairs = [(int(c) // len(self.chunks), int(c) % len(self.chunks)) for c in codes.cpu()]
        self.slots = torch.full((size, size), -1, device=device, dtype=torch.long)
        self.slots[eligible] = torch.searchsorted(codes, pairs[eligible])
        self.eligible = eligible
        self.fixed_on = self.causal & ~eligible
        if condition.endswith('_off'):
            self.fixed_on &= ~mutable
        self.count = len(self.pairs)
        self.keep = math.floor(.2 * self.count + .5)
        self.device = torch.device(device)
        assert self.count > 0 and self.fixed_on.diagonal().all()

    def additive(self, gates):
        assert gates.shape == (self.count,)
        weights = torch.where(self.eligible, gates[self.slots.clamp_min(0)], self.fixed_on.float())
        # Zero is exactly blocked; clamping only inside the unselected logarithm avoids NaN gradients.
        logged = torch.log(weights.clamp_min(torch.finfo(torch.float32).tiny))
        return torch.where(weights > 0, logged, float('-inf'))[None, None]

    def random(self, seed):
        generator = torch.Generator(device='cpu').manual_seed(seed)
        chosen = torch.randperm(self.count, generator=generator)[:self.keep].to(self.device)
        return torch.zeros(self.count, device=self.device).scatter_(0, chosen, 1.)

    def hard(self, alpha):
        order = torch.argsort(alpha.detach(), descending=True, stable=True)
        return torch.zeros_like(alpha).scatter_(0, order[:self.keep], 1.)

    def structural_zero_cells(self):
        # No prediction after the final reasoning query contributes to the trace loss.
        active = self.slots[:self.length - 1]
        return sorted(set(range(self.count)) - set(active[active >= 0].cpu().tolist()))


class AttentionMask:
    """One shared additive bias at every layer/head; all softmaxes accumulate in FP32."""
    def __init__(self, model):
        self.bias = None
        self.original = []
        for layer in model.model.layers:
            module = layer.self_attn
            self.original.append((module, module.forward))
            controller = self

            def forward(this, hidden_states, position_embeddings, attention_mask=None,
                        past_key_values=None, cache_position=None, **kwargs):
                assert past_key_values is None, 'Pilot replay never uses a KV cache'
                batch, length, _ = hidden_states.shape
                shape = (batch, length, -1, this.head_dim)
                q = this.q_norm(this.q_proj(hidden_states).view(shape)).transpose(1, 2)
                k = this.k_norm(this.k_proj(hidden_states).view(shape)).transpose(1, 2)
                v = this.v_proj(hidden_states).view(shape).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
                k = repeat_kv(k, this.num_key_value_groups)
                v = repeat_kv(v, this.num_key_value_groups)
                scores = (q @ k.transpose(2, 3)) * this.scaling
                scores = scores.float()
                if controller.bias is None:
                    # Match unmodified eager attention, including the native causal mask.
                    if attention_mask is not None:
                        scores = scores + attention_mask[:, :, :, :length]
                    else:
                        future = torch.ones(length, length, device=q.device, dtype=torch.bool).triu(1)
                        scores = scores.masked_fill(future, float('-inf'))
                else:
                    assert controller.bias.shape[-2:] == (length, length)
                    scores = scores + controller.bias.to(q.device)
                probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                output = (probabilities @ v).transpose(1, 2).contiguous().reshape(batch, length, -1)
                return this.o_proj(output), None

            module.forward = types.MethodType(forward, module)

    def restore(self):
        for module, fn in self.original:
            module.forward = fn


def trace_nll(model, ids, prompt_length, block_size=64):
    hidden = model.model(ids, use_cache=False).last_hidden_state[:, prompt_length - 1:-1]
    targets = ids[:, prompt_length:]
    assert hidden.shape[1] == targets.shape[1] > 0
    total = hidden.new_zeros((), dtype=torch.float32)

    def block_loss(h, y):
        logits = model.lm_head(h).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction='sum')

    for start in range(0, targets.shape[1], block_size):
        h, y = hidden[:, start:start+block_size], targets[:, start:start+block_size]
        if torch.is_grad_enabled():
            total = total + checkpoint(block_loss, h, y, use_reentrant=False)
        else:
            total = total + block_loss(h, y)
    return total / targets.numel()


def sample_gates(alpha):
    u = torch.empty_like(alpha).uniform_(1e-6, 1-1e-6)
    v = torch.sigmoid((torch.log(u) - torch.log1p(-u) + alpha) / (2/3))
    return (1.2 * v - .1).clamp(0, 1)


def size_loss(alpha):
    expected = torch.sigmoid(alpha - (2/3) * math.log(.1/1.1)).sum()
    return (expected - .2 * alpha.numel()).square() / alpha.numel()


def size_coefficient(step):
    fraction = step / 999
    return 0. if fraction < .25 else 1000. * min(1., (fraction - .25) / .5)
