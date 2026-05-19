import argparse
from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 1024,
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "d_ff": 2048,
        "lang_src": "en",
        "lang_tgt": "pa",
        "datasource": "samanantar",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "beam_size": 4,
    }

def get_config_from_args():
    defaults = get_config()
    parser = argparse.ArgumentParser()
    for key, val in defaults.items():
        if val is None:
            parser.add_argument(f"--{key}", default=val)
        else:
            parser.add_argument(f"--{key}", type=type(val), default=val)
    args = parser.parse_args()
    return vars(args)

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)