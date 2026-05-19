import sys
from unittest.mock import patch

from config import get_config, get_config_from_args, get_weights_file_path


class TestGetConfig:
    def test_returns_dict(self):
        assert isinstance(get_config(), dict)

    def test_required_keys_present(self):
        required = {
            "batch_size", "num_epochs", "lr", "seq_len", "d_model",
            "num_layers", "num_heads", "dropout", "d_ff",
            "lang_src", "lang_tgt", "model_folder", "model_basename",
            "tokenizer_file", "experiment_name", "beam_size",
        }
        config = get_config()
        missing = required - set(config.keys())
        assert not missing, f"Missing config keys: {missing}"

    def test_d_model_divisible_by_num_heads(self):
        c = get_config()
        assert c["d_model"] % c["num_heads"] == 0, \
            "d_model must be divisible by num_heads for multi-head attention"

    def test_lr_is_positive(self):
        assert get_config()["lr"] > 0

    def test_seq_len_is_positive(self):
        assert get_config()["seq_len"] > 0


class TestGetConfigFromArgs:
    def test_defaults_match_get_config(self):
        with patch.object(sys, "argv", ["train.py"]):
            cli_config = get_config_from_args()
        default_config = get_config()
        for key in default_config:
            assert cli_config[key] == default_config[key], \
                f"Mismatch for key '{key}': {cli_config[key]} vs {default_config[key]}"

    def test_cli_override_batch_size(self):
        with patch.object(sys, "argv", ["train.py", "--batch_size", "32"]):
            config = get_config_from_args()
        assert config["batch_size"] == 32

    def test_cli_override_lr(self):
        with patch.object(sys, "argv", ["train.py", "--lr", "5e-4"]):
            config = get_config_from_args()
        assert abs(config["lr"] - 5e-4) < 1e-10

    def test_cli_override_num_layers(self):
        with patch.object(sys, "argv", ["train.py", "--num_layers", "4"]):
            config = get_config_from_args()
        assert config["num_layers"] == 4

    def test_non_overridden_keys_keep_defaults(self):
        with patch.object(sys, "argv", ["train.py", "--batch_size", "16"]):
            config = get_config_from_args()
        assert config["num_epochs"] == get_config()["num_epochs"]


class TestGetWeightsFilePath:
    def test_returns_string(self):
        path = get_weights_file_path(get_config(), "05")
        assert isinstance(path, str)

    def test_contains_epoch(self):
        path = get_weights_file_path(get_config(), "07")
        assert "07" in path

    def test_ends_with_pt(self):
        path = get_weights_file_path(get_config(), "03")
        assert path.endswith(".pt")

    def test_contains_datasource(self):
        config = get_config()
        path = get_weights_file_path(config, "01")
        assert config["datasource"] in path
