import argparse

import numpy
import torch
from absl.testing import absltest, parameterized
from src.modules.structformer_layer import (
    StructformerDecoderLayer,
    StructformerEncoderLayer,
)


class StructformerDecoderLayerTest(parameterized.TestCase):
    @parameterized.parameters({"constrain": False}, {"constrain": True})
    def test_forward(self, constrain):
        batch_size = 5
        seq_len = 7
        embed_dim = 10
        ffn_embed_dim = 10
        num_heads = 2

        overrides = {
            "decoder_embed_dim": embed_dim,
            "decoder_ffn_embed_dim": ffn_embed_dim,
            "decoder_attention_heads": num_heads,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "decoder_normalize_before": False,
            # options for structformer
            "induction_layer": 1,
            "induction_layer_connect": False,
            "gate_attention_heads": 2,
            "head_temperature": 1.0,
            "head_activation_fn": "linear",
        }
        args = argparse.Namespace(**overrides)

        # no_encoder_attn=True because self-attention is only gated and it is unnessesary to check the beheivar of encoder-decoder attention.
        layer = StructformerDecoderLayer(args, no_encoder_attn=True)

        # `self_attn_gate` is a tensor with positive probabilities.
        self_attn_gate = (
            torch.rand(batch_size * num_heads, seq_len, seq_len) if constrain else None
        )

        x = torch.randn(seq_len, batch_size, embed_dim)

        x, attn, _ = layer(x, self_attn_gate=self_attn_gate, need_attn=True)

        self.assertEqual(list(x.shape), [seq_len, batch_size, embed_dim])
        self.assertEqual(list(attn.shape), [batch_size, seq_len, seq_len])

        numpy.testing.assert_allclose(
            attn.sum(-1).detach().numpy(),
            numpy.ones((batch_size, seq_len)),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    absltest.main()
