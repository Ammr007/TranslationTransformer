import pytest
import torch
from unittest.mock import MagicMock

from model import build_transformer


D_MODEL = 32
SEQ_LEN = 16
BATCH_SIZE = 2
NUM_HEADS = 4
D_FF = 64
NUM_LAYERS = 2
DROPOUT = 0.0   
VOCAB_SIZE = 100


@pytest.fixture
def dims():
    """Return a dict of shared dimensions so tests don't hardcode magic numbers."""
    return {
        "d_model":    D_MODEL,
        "seq_len":    SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "num_heads":  NUM_HEADS,
        "d_ff":       D_FF,
        "num_layers": NUM_LAYERS,
        "vocab_size": VOCAB_SIZE,
    }


@pytest.fixture
def transformer(dims):
    """Full Transformer with tiny dims, eval mode, no dropout."""
    model = build_transformer(
        src_vocab_size=dims["vocab_size"],
        tgt_vocab_size=dims["vocab_size"],
        src_seq_len=dims["seq_len"],
        tgt_seq_len=dims["seq_len"],
        d_model=dims["d_model"],
        n=dims["num_layers"],
        h=dims["num_heads"],
        dropout=DROPOUT,
        d_ff=dims["d_ff"],
    )
    model.eval()
    return model


@pytest.fixture
def src_batch(dims):
    """Random integer token ids for a source batch."""
    return torch.randint(0, dims["vocab_size"], (dims["batch_size"], dims["seq_len"]))


@pytest.fixture
def tgt_batch(dims):
    """Random integer token ids for a target batch."""
    return torch.randint(0, dims["vocab_size"], (dims["batch_size"], dims["seq_len"]))


@pytest.fixture
def src_mask(dims):
    """All-ones encoder mask (no padding) — shape (batch, 1, 1, seq)."""
    return torch.ones(dims["batch_size"], 1, 1, dims["seq_len"]).int()


@pytest.fixture
def tgt_mask(dims):
    """Causal decoder mask — shape (batch, 1, seq, seq)."""
    from dataset import causal_mask
    mask = causal_mask(dims["seq_len"])                        # (1, seq, seq)
    return mask.unsqueeze(0).expand(dims["batch_size"], -1, -1, -1)


@pytest.fixture
def mock_tokenizer():
    """Minimal tokenizer mock for inference and dataset tests."""
    tok = MagicMock()
    tok.token_to_id.side_effect = lambda t: {"[SOS]": 1, "[EOS]": 2, "[PAD]": 0, "[UNK]": 3}[t]
    tok.get_vocab_size.return_value = VOCAB_SIZE
    enc = MagicMock()
    enc.ids = [10, 11, 12]
    tok.encode.return_value = enc
    tok.decode.return_value = "ਸਤ ਸ੍ਰੀ ਅਕਾਲ"
    return tok
