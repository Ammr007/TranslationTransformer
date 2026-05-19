import math
import torch
import torch.nn as nn
import pytest

from model import (
    InputEmbedding,
    PositionalEncoding,
    LayerNormalization,
    FeedForward,
    MultiHeadAttentionBlock,
    LinearProjectionLayer,
    build_transformer,
)

def assert_no_nan_inf(tensor: torch.Tensor, name: str = "tensor"):
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"

class TestInputEmbedding:
    def test_output_shape(self, dims):
        emb = InputEmbedding(dims["d_model"], dims["vocab_size"])
        x   = torch.randint(0, dims["vocab_size"], (dims["batch_size"], dims["seq_len"]))
        out = emb(x)
        assert out.shape == (dims["batch_size"], dims["seq_len"], dims["d_model"])

    def test_scaling(self, dims):
        """Embedding vectors are scaled by √d_model."""
        emb = InputEmbedding(dims["d_model"], dims["vocab_size"])
        x   = torch.zeros(1, 1, dtype=torch.long)   # single token
        # raw embedding weight for token 0
        raw = emb.embedding(x)
        out = emb(x)
        expected_scale = math.sqrt(dims["d_model"])
        torch.testing.assert_close(out, raw * expected_scale)

    def test_no_nan(self, dims):
        emb = InputEmbedding(dims["d_model"], dims["vocab_size"])
        x   = torch.randint(0, dims["vocab_size"], (dims["batch_size"], dims["seq_len"]))
        assert_no_nan_inf(emb(x), "InputEmbedding output")

class TestPositionalEncoding:
    def test_output_shape(self, dims):
        pe  = PositionalEncoding(dims["d_model"], dims["seq_len"], dropout=0.0)
        x   = torch.zeros(dims["batch_size"], dims["seq_len"], dims["d_model"])
        out = pe(x)
        assert out.shape == x.shape

    def test_batch_size_invariant(self, dims):
        """Output shape must be correct regardless of batch size (bug regression)."""
        pe = PositionalEncoding(dims["d_model"], dims["seq_len"], dropout=0.0)
        for batch in [1, 2, 8]:
            x   = torch.zeros(batch, dims["seq_len"], dims["d_model"])
            out = pe(x)
            assert out.shape == (batch, dims["seq_len"], dims["d_model"]), \
                f"Failed for batch_size={batch}"

    def test_pe_is_not_trainable(self, dims):
        pe = PositionalEncoding(dims["d_model"], dims["seq_len"], dropout=0.0)
        # pe buffer should exist but NOT appear in parameters()
        param_names = [n for n, _ in pe.named_parameters()]
        assert "pe" not in param_names

    def test_no_nan(self, dims):
        pe  = PositionalEncoding(dims["d_model"], dims["seq_len"], dropout=0.0)
        x   = torch.randn(dims["batch_size"], dims["seq_len"], dims["d_model"])
        assert_no_nan_inf(pe(x), "PositionalEncoding output")

class TestLayerNormalization:
    def test_output_shape(self, dims):
        norm = LayerNormalization(dims["d_model"])
        x    = torch.randn(dims["batch_size"], dims["seq_len"], dims["d_model"])
        assert norm(x).shape == x.shape

    def test_normalises_mean_and_std(self, dims):
        """After normalisation (with default alpha=1, bias=0) the output should
        have mean ≈ 0 and std ≈ 1 along the last dimension."""
        norm = LayerNormalization(dims["d_model"])
        # reset to identity: alpha=1, bias=0
        nn.init.ones_(norm.alpha)
        nn.init.zeros_(norm.bias)
        x   = torch.randn(4, 8, dims["d_model"]) * 10 + 5   # deliberately off-scale
        out = norm(x)
        assert out.mean().abs() < 0.1,  "Mean should be ≈ 0 after LayerNorm"
        assert (out.std() - 1.0).abs() < 0.1, "Std should be ≈ 1 after LayerNorm"

    def test_per_feature_params(self, dims):
        """alpha and bias must be vectors of length d_model, not scalars."""
        norm = LayerNormalization(dims["d_model"])
        assert norm.alpha.shape == (dims["d_model"],)
        assert norm.bias.shape  == (dims["d_model"],)

    def test_no_nan_on_constant_input(self, dims):
        """Constant input would give std=0; eps must prevent division by zero."""
        norm = LayerNormalization(dims["d_model"])
        x    = torch.ones(2, 4, dims["d_model"])
        assert_no_nan_inf(norm(x), "LayerNorm output on constant input")

class TestFeedForward:
    def test_output_shape(self, dims):
        ff  = FeedForward(dims["d_model"], dims["d_ff"], dropout=0.0)
        x   = torch.randn(dims["batch_size"], dims["seq_len"], dims["d_model"])
        assert ff(x).shape == x.shape

    def test_no_nan(self, dims):
        ff  = FeedForward(dims["d_model"], dims["d_ff"], dropout=0.0)
        x   = torch.randn(dims["batch_size"], dims["seq_len"], dims["d_model"])
        assert_no_nan_inf(ff(x), "FeedForward output")

class TestMultiHeadAttention:
    def test_self_attention_shape(self, dims):
        attn = MultiHeadAttentionBlock(dims["d_model"], dims["num_heads"], dropout=0.0)
        x    = torch.randn(dims["batch_size"], dims["seq_len"], dims["d_model"])
        mask = torch.ones(dims["batch_size"], 1, dims["seq_len"], dims["seq_len"])
        out  = attn(x, x, x, mask)
        assert out.shape == x.shape

    def test_cross_attention_shape(self, dims):
        """Decoder cross-attention: query from target, key/value from encoder output."""
        attn    = MultiHeadAttentionBlock(dims["d_model"], dims["num_heads"], dropout=0.0)
        query   = torch.randn(dims["batch_size"], dims["seq_len"], dims["d_model"])
        enc_out = torch.randn(dims["batch_size"], dims["seq_len"], dims["d_model"])
        mask    = torch.ones(dims["batch_size"], 1, dims["seq_len"], dims["seq_len"])
        out     = attn(query, enc_out, enc_out, mask)
        assert out.shape == query.shape

    def test_causal_mask_prevents_future_access(self, dims):
        """
        With a strict causal mask the output at position t should not depend on
        any input at position > t.  We verify this by checking that perturbing
        a future token leaves past outputs unchanged.
        """
        attn = MultiHeadAttentionBlock(dims["d_model"], dims["num_heads"], dropout=0.0)
        attn.eval()

        from dataset import causal_mask
        seq = dims["seq_len"]
        mask = causal_mask(seq).unsqueeze(0)  # (1,1,seq,seq) — broadcast over batch

        x        = torch.randn(1, seq, dims["d_model"])
        x_perturb = x.clone()
        x_perturb[0, seq // 2 :, :] += 100.0   # corrupt second half

        with torch.no_grad():
            out        = attn(x,         x,         x,         mask)
            out_perturb = attn(x_perturb, x_perturb, x_perturb, mask)

        # Outputs for positions BEFORE the perturbation must be identical
        torch.testing.assert_close(out[0, : seq // 2], out_perturb[0, : seq // 2])

    def test_no_nan(self, dims):
        attn = MultiHeadAttentionBlock(dims["d_model"], dims["num_heads"], dropout=0.0)
        x    = torch.randn(dims["batch_size"], dims["seq_len"], dims["d_model"])
        mask = torch.ones(dims["batch_size"], 1, dims["seq_len"], dims["seq_len"])
        assert_no_nan_inf(attn(x, x, x, mask), "MHA output")

    def test_d_model_not_divisible_by_heads_raises(self, dims):
        with pytest.raises(AssertionError):
            MultiHeadAttentionBlock(d_model=33, h=8, dropout=0.0)

class TestEncoder:
    def test_output_shape(self, transformer, src_batch, src_mask, dims):
        src = transformer.src_embed(src_batch)
        src = transformer.src_pos(src)
        out = transformer.encoder(src, src_mask)
        assert out.shape == (dims["batch_size"], dims["seq_len"], dims["d_model"])

    def test_no_nan(self, transformer, src_batch, src_mask):
        src = transformer.src_embed(src_batch)
        src = transformer.src_pos(src)
        assert_no_nan_inf(transformer.encoder(src, src_mask), "Encoder output")

class TestDecoder:
    def test_output_shape(self, transformer, src_batch, tgt_batch,
                          src_mask, tgt_mask, dims):
        enc_out = transformer.encode(src_batch, src_mask)
        tgt     = transformer.tgt_embed(tgt_batch)
        tgt     = transformer.tgt_pos(tgt)
        out     = transformer.decoder(tgt, enc_out, src_mask, tgt_mask)
        assert out.shape == (dims["batch_size"], dims["seq_len"], dims["d_model"])

    def test_no_nan(self, transformer, src_batch, tgt_batch, src_mask, tgt_mask):
        enc_out = transformer.encode(src_batch, src_mask)
        tgt     = transformer.tgt_embed(tgt_batch)
        tgt     = transformer.tgt_pos(tgt)
        assert_no_nan_inf(
            transformer.decoder(tgt, enc_out, src_mask, tgt_mask),
            "Decoder output",
        )

class TestLinearProjectionLayer:
    def test_output_shape(self, dims):
        proj = LinearProjectionLayer(dims["d_model"], dims["vocab_size"])
        x    = torch.randn(dims["batch_size"], dims["seq_len"], dims["d_model"])
        out  = proj(x)
        assert out.shape == (dims["batch_size"], dims["seq_len"], dims["vocab_size"])

    def test_output_is_log_prob(self, dims):
        """Values must be ≤ 0 (log of probability ≤ 1) and sum to 0 in log-space."""
        proj = LinearProjectionLayer(dims["d_model"], dims["vocab_size"])
        x    = torch.randn(2, 4, dims["d_model"])
        out  = proj(x)
        assert (out <= 0).all(), "log_softmax outputs should be ≤ 0"
        # exp(log_softmax) should sum to 1 along vocab dim
        probs = out.exp()
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))

class TestTransformer:
    def test_encode_shape(self, transformer, src_batch, src_mask, dims):
        out = transformer.encode(src_batch, src_mask)
        assert out.shape == (dims["batch_size"], dims["seq_len"], dims["d_model"])

    def test_decode_shape(self, transformer, src_batch, tgt_batch,
                          src_mask, tgt_mask, dims):
        enc_out = transformer.encode(src_batch, src_mask)
        out     = transformer.decode(enc_out, src_mask, tgt_batch, tgt_mask)
        assert out.shape == (dims["batch_size"], dims["seq_len"], dims["d_model"])

    def test_project_shape(self, transformer, src_batch, tgt_batch,
                           src_mask, tgt_mask, dims):
        enc_out  = transformer.encode(src_batch, src_mask)
        dec_out  = transformer.decode(enc_out, src_mask, tgt_batch, tgt_mask)
        proj_out = transformer.project(dec_out)
        assert proj_out.shape == (dims["batch_size"], dims["seq_len"], dims["vocab_size"])

    def test_full_forward_no_nan(self, transformer, src_batch, tgt_batch,
                                  src_mask, tgt_mask):
        enc_out  = transformer.encode(src_batch, src_mask)
        dec_out  = transformer.decode(enc_out, src_mask, tgt_batch, tgt_mask)
        proj_out = transformer.project(dec_out)
        assert_no_nan_inf(proj_out, "full forward pass output")

    def test_build_transformer_layer_count(self, dims):
        """build_transformer must create exactly n encoder and n decoder blocks (bug regression)."""
        n_layers = 3
        model = build_transformer(
            src_vocab_size=dims["vocab_size"],
            tgt_vocab_size=dims["vocab_size"],
            src_seq_len=dims["seq_len"],
            tgt_seq_len=dims["seq_len"],
            d_model=dims["d_model"],
            n=n_layers,
            h=dims["num_heads"],
        )
        assert len(model.encoder.layers) == n_layers, \
            "Encoder layer count does not match n"
        assert len(model.decoder.layers) == n_layers, \
            "Decoder layer count does not match n"
