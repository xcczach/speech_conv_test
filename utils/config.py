import json
import os
from huggingface_hub import snapshot_download

CKPT_DIR = "ckpts"
def download_models(names: list[str]|None=None, token: str|None=None):
    with open('config.json', 'r') as f:
        config = json.load(f)
    models = config['models']
    if names is not None:
        models = {k: v for k, v in models.items() if k in names}
    os.makedirs(CKPT_DIR,exist_ok=True)
    for name, path in models.items():
        model_path = os.path.join(CKPT_DIR, name)
        os.makedirs(model_path, exist_ok=True)
        print(f"Downloading {name} from {path}")
        snapshot_download(repo_id=path, local_dir=model_path, token=token)

def list_models():
    model_dirs = os.listdir(CKPT_DIR)
    return model_dirs

def get_model_path(name: str):
    return os.path.join(CKPT_DIR, name)