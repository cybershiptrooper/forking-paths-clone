import copy
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from expts.prompt_bias_circuit_discovery.monitor_pilot_0924.core import (
    CellPool, AttentionMask, trace_nll, sample_gates, size_coefficient,
)


def record():
    return dict(input_ids=list(range(12)), prompt_length=6, chunks=[
        dict(id='P00', start=0, end=2), dict(id='P01', start=3, end=5),
        dict(id='R01', start=6, end=8), dict(id='R02', start=9, end=11)])


def tiny():
    torch.manual_seed(73)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, attention_dropout=0., max_position_embeddings=64)
    config._attn_implementation = 'eager'
    config.use_cache = False
    return Qwen3ForCausalLM(config).eval()


class PilotTests(unittest.TestCase):
    def test_scopes_exact_zeros_and_probe_reads(self):
        for cond in ['rr_on', 'rr_off', 'pr_on', 'pr_off', 'joint']:
            p = CellPool(record(), cond, with_probe=True)
            a = p.additive(torch.zeros(p.count))[0, 0]
            self.assertTrue((a.diagonal() == 0).all())
            self.assertTrue(torch.isneginf(a[torch.ones_like(a, dtype=torch.bool).triu(1)]).all())
            self.assertTrue(torch.isneginf(a[p.eligible]).all())
            self.assertTrue((a[12, :13] == 0).all())
            for q in range(18):
                self.assertTrue((a[q, :min(3, q+1)] == 0).all())
            if cond.endswith('_off'):
                self.assertTrue(torch.isneginf(a[5, 3]))
            if cond.startswith('rr'):
                self.assertTrue(p.eligible[8, 6])  # same chunk, earlier token
                self.assertFalse(p.eligible[6, 5])
            if cond.startswith('pr'):
                self.assertTrue(p.eligible[6, 5])
                self.assertFalse(p.eligible[8, 6])

    def test_all_one_identity_and_loss_alignment(self):
        m = tiny()
        ids = torch.tensor([record()['input_ids']])
        with torch.no_grad():
            reference = m(ids, use_cache=False).logits
            direct = torch.nn.functional.cross_entropy(reference[:, 5:-1].reshape(-1, 32), ids[:, 6:].reshape(-1))
        controller = AttentionMask(m)
        for cond in ['rr_on', 'pr_on', 'joint']:
            p = CellPool(record(), cond)
            controller.bias = p.additive(torch.ones(p.count))
            with torch.no_grad():
                actual = m(ids, use_cache=False).logits
                loss = trace_nll(m, ids, 6, block_size=2)
            torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(loss, direct, atol=1e-7, rtol=1e-6)
        controller.restore()

    def test_gate_gradients_checkpointing_and_finite_difference(self):
        m = tiny()
        for param in m.parameters():
            param.requires_grad_(False)
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        m.enable_input_require_grads()
        m.train()
        controller = AttentionMask(m)
        pool = CellPool(record(), 'joint')
        ids = torch.tensor([record()['input_ids']])
        gates = torch.full((pool.count,), .55, requires_grad=True)
        controller.bias = pool.additive(gates)
        loss = trace_nll(m, ids, 6, block_size=2)
        loss.backward()
        self.assertTrue(torch.isfinite(gates.grad).all())
        index = int(gates.grad.abs().argmax())
        self.assertGreater(float(gates.grad[index].abs()), 1e-7)
        values = []
        with torch.no_grad():
            for sign in [-1, 1]:
                perturbed = gates.detach().clone()
                perturbed[index] += sign * .02
                controller.bias = pool.additive(perturbed)
                values.append(float(trace_nll(m, ids, 6)))
        estimate = (values[1] - values[0]) / .04
        self.assertAlmostEqual(float(gates.grad[index]), estimate, delta=2e-5)
        # Truly zero gates are blocked without NaN derivatives.
        gates = torch.zeros(pool.count, requires_grad=True)
        controller.bias = pool.additive(gates)
        trace_nll(m, ids, 6).backward()
        self.assertTrue(torch.isfinite(gates.grad).all())

    def test_no_future_target_leakage(self):
        m = tiny()
        controller = AttentionMask(m)
        pool = CellPool(record(), 'joint')
        controller.bias = pool.additive(torch.ones(pool.count))
        ids = torch.tensor([record()['input_ids']])
        changed = ids.clone(); changed[0, 9:] = 23
        with torch.no_grad():
            a, b = m(ids).logits, m(changed).logits
        torch.testing.assert_close(a[:, :9], b[:, :9], atol=0, rtol=0)

    def test_first_reasoning_prediction_constant_for_rr_and_pr(self):
        m = tiny(); controller = AttentionMask(m)
        ids = torch.tensor([record()['input_ids']])
        for cond in ['rr_on', 'rr_off', 'pr_on', 'pr_off']:
            pool = CellPool(record(), cond)
            outputs = []
            for value in [0., 1.]:
                controller.bias = pool.additive(torch.full((pool.count,), value))
                with torch.no_grad():
                    outputs.append(m(ids).logits[:, 5])
            torch.testing.assert_close(*outputs, atol=0, rtol=0)

    def test_exact_retention_and_tie_order(self):
        for cond in ['rr_on', 'rr_off', 'pr_on', 'pr_off', 'joint']:
            pool = CellPool(record(), cond)
            binary = pool.hard(torch.zeros(pool.count))
            self.assertEqual(int(binary.sum()), pool.keep)
            self.assertTrue((binary[:pool.keep] == 1).all())
            self.assertEqual(int(pool.random(0).sum()), pool.keep)
        self.assertEqual(size_coefficient(0), 0)
        self.assertEqual(size_coefficient(999), 1000)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
