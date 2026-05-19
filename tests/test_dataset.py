import torch
import pytest
from unittest.mock import MagicMock

from dataset import LanguageDataset, causal_mask


def make_tokenizer(pad_id=0, sos_id=1, eos_id=2, encode_ids=None):
    """Return a mock tokenizer that encodes any string to `encode_ids`."""
    tok = MagicMock()
    tok.token_to_id.side_effect = lambda t: {
        "[PAD]": pad_id, "[SOS]": sos_id, "[EOS]": eos_id,
    }[t]
    enc = MagicMock()
    enc.ids = encode_ids if encode_ids is not None else [10, 11, 12]
    tok.encode.return_value = enc
    return tok


def make_dataset(seq_len=20, src_len=5, tgt_len=5):
    """Construct a single-item LanguageDataset with controllable token lengths."""
    tokenizer_src = make_tokenizer(encode_ids=list(range(10, 10 + src_len)))
    tokenizer_tgt = make_tokenizer(encode_ids=list(range(20, 20 + tgt_len)))
    ds = [{"src": "hello world", "tgt": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ"}]
    return LanguageDataset(ds, tokenizer_src, tokenizer_tgt, seq_len)


class TestCausalMask:
    def test_shape(self):
        mask = causal_mask(8)
        assert mask.shape == (1, 8, 8)

    def test_lower_triangular(self):
        """Position i should attend to positions 0..i only (True on/below diagonal)."""
        mask = causal_mask(6).squeeze(0)
        for i in range(6):
            for j in range(6):
                expected = j <= i
                assert mask[i, j].item() == expected, \
                    f"mask[{i},{j}] expected {expected}, got {mask[i,j].item()}"

    def test_dtype_is_bool_compatible(self):
        mask = causal_mask(4)
        # Must be usable as a boolean-style mask (0/1 int or bool)
        assert mask.dtype in (torch.bool, torch.int, torch.int64)

class TestLanguageDataset:
    def test_len(self):
        ds_raw = [{"src": "a", "tgt": "b"}, {"src": "c", "tgt": "d"}]
        tok_src = make_tokenizer(encode_ids=[5])
        tok_tgt = make_tokenizer(encode_ids=[6])
        ds = LanguageDataset(ds_raw, tok_src, tok_tgt, seq_len=10)
        assert len(ds) == 2

    def test_output_keys(self):
        ds  = make_dataset()
        item = ds[0]
        expected_keys = {
            "encoder_input", "decoder_input", "encoder_mask",
            "decoder_mask", "label", "src_text", "tgt_text",
        }
        assert set(item.keys()) == expected_keys

    def test_encoder_input_shape(self):
        seq_len = 20
        ds   = make_dataset(seq_len=seq_len, src_len=5)
        item = ds[0]
        assert item["encoder_input"].shape == (seq_len,)

    def test_decoder_input_shape(self):
        seq_len = 20
        ds   = make_dataset(seq_len=seq_len, tgt_len=5)
        item = ds[0]
        assert item["decoder_input"].shape == (seq_len,)

    def test_label_shape(self):
        seq_len = 20
        ds   = make_dataset(seq_len=seq_len, tgt_len=5)
        item = ds[0]
        assert item["label"].shape == (seq_len,)

    def test_encoder_input_starts_with_sos(self):
        """Encoder input: [SOS] token_ids... [EOS] [PAD]..."""
        ds   = make_dataset()
        item = ds[0]
        assert item["encoder_input"][0].item() == 1   # SOS id

    def test_encoder_input_ends_with_eos_then_pad(self):
        seq_len = 20
        src_len = 5
        ds   = make_dataset(seq_len=seq_len, src_len=src_len)
        item = ds[0]
        eos_pos = src_len + 1          # SOS + 5 tokens
        assert item["encoder_input"][eos_pos].item() == 2   # EOS id
        # Everything after EOS should be PAD (id=0)
        assert (item["encoder_input"][eos_pos + 1:] == 0).all()

    def test_decoder_input_starts_with_sos_no_eos(self):
        """Decoder input: [SOS] token_ids... [PAD]... (no EOS — that's in the label)."""
        ds   = make_dataset()
        item = ds[0]
        assert item["decoder_input"][0].item() == 1   # SOS
        assert 2 not in item["decoder_input"].tolist(), \
            "Decoder input should not contain EOS"

    def test_label_ends_with_eos(self):
        """Label: token_ids... [EOS] [PAD]..."""
        seq_len = 20
        tgt_len = 5
        ds   = make_dataset(seq_len=seq_len, tgt_len=tgt_len)
        item = ds[0]
        eos_pos = tgt_len              # 5 tokens then EOS
        assert item["label"][eos_pos].item() == 2   # EOS id

    def test_decoder_input_and_label_are_offset_by_one(self):
        """
        label[t] should equal decoder_input[t+1] for the non-padding region.
        This is the teacher-forcing offset.
        """
        seq_len = 20
        tgt_len = 4
        ds   = make_dataset(seq_len=seq_len, tgt_len=tgt_len)
        item = ds[0]
        # decoder_input: [SOS, t0, t1, t2, t3, PAD...]
        # label:         [t0, t1, t2, t3, EOS, PAD...]
        dec = item["decoder_input"]
        lbl = item["label"]
        for i in range(tgt_len):
            assert lbl[i].item() == dec[i + 1].item(), \
                f"label[{i}]={lbl[i]} != decoder_input[{i+1}]={dec[i+1]}"

    def test_encoder_mask_shape(self):
        seq_len = 20
        ds   = make_dataset(seq_len=seq_len)
        item = ds[0]
        # (1, 1, seq_len) — will broadcast over batch and heads
        assert item["encoder_mask"].shape == (1, 1, seq_len)

    def test_decoder_mask_shape(self):
        seq_len = 20
        ds   = make_dataset(seq_len=seq_len)
        item = ds[0]
        # (1, seq_len, seq_len)
        assert item["decoder_mask"].shape == (1, seq_len, seq_len)

    def test_too_long_raises_value_error(self):
        """Sentences longer than seq_len - 2 must raise ValueError, not silently truncate."""
        tok_src = make_tokenizer(encode_ids=list(range(50)))  # 50 tokens
        tok_tgt = make_tokenizer(encode_ids=list(range(5)))
        ds = LanguageDataset([{"src": "x", "tgt": "y"}], tok_src, tok_tgt, seq_len=10)
        with pytest.raises(ValueError, match="too long"):
            _ = ds[0]

    def test_pad_token_stored_as_int(self):
        """Regression: pad_token_id must be a plain int, not a tensor."""
        ds = make_dataset()
        assert isinstance(ds.pad_token_id, int), \
            "pad_token_id should be int, not tensor (causes incorrect padding)"
