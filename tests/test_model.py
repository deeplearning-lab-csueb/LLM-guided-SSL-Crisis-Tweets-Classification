"""Tests for model.py — requires torch + transformers (skips if unavailable)."""

import sys
import unittest

sys.path.insert(0, "/workspace")


def _requires_torch(test_func):
    """Decorator to skip tests if torch/transformers are unavailable."""
    def wrapper(self):
        try:
            import torch
            import transformers
        except ImportError:
            self.skipTest("torch/transformers not available")
        return test_func(self)
    wrapper.__name__ = test_func.__name__
    wrapper.__doc__ = test_func.__doc__
    return wrapper


class TestBertClassifierForward(unittest.TestCase):
    """BertClassifier forward pass shape tests."""

    @_requires_torch
    def test_output_shape(self):
        from lg_cotrain.model import create_fresh_model
        from lg_cotrain.config import LGCoTrainConfig
        import torch
        cfg = LGCoTrainConfig(model_name="prajjwal1/bert-tiny", num_labels=10)
        model = create_fresh_model(cfg)
        input_ids = torch.randint(0, 1000, (4, 16))
        attention_mask = torch.ones(4, 16, dtype=torch.long)
        logits = model(input_ids, attention_mask)
        self.assertEqual(logits.shape, (4, 10))

    @_requires_torch
    def test_predict_proba_sums_to_one(self):
        from lg_cotrain.model import create_fresh_model
        from lg_cotrain.config import LGCoTrainConfig
        import torch
        cfg = LGCoTrainConfig(model_name="prajjwal1/bert-tiny", num_labels=10)
        model = create_fresh_model(cfg)
        model.eval()
        input_ids = torch.randint(0, 1000, (4, 16))
        attention_mask = torch.ones(4, 16, dtype=torch.long)
        probs = model.predict_proba(input_ids, attention_mask)
        self.assertEqual(probs.shape, (4, 10))
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)

    @_requires_torch
    def test_predict_proba_no_grad(self):
        from lg_cotrain.model import create_fresh_model
        from lg_cotrain.config import LGCoTrainConfig
        import torch
        cfg = LGCoTrainConfig(model_name="prajjwal1/bert-tiny", num_labels=10)
        model = create_fresh_model(cfg)
        model.eval()
        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        probs = model.predict_proba(input_ids, attention_mask)
        self.assertFalse(probs.requires_grad)

    @_requires_torch
    def test_create_fresh_returns_new_instance(self):
        from lg_cotrain.model import create_fresh_model
        from lg_cotrain.config import LGCoTrainConfig
        cfg = LGCoTrainConfig(model_name="prajjwal1/bert-tiny", num_labels=10)
        m1 = create_fresh_model(cfg)
        m2 = create_fresh_model(cfg)
        self.assertIsNot(m1, m2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
