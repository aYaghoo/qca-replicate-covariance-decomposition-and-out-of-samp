import os
import subprocess

import datasets
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tensorflow import keras
from transformers import AutoModel

# ruleid: torch-load-without-weights-only
torch.load("m.pt")
# ok: torch-load-without-weights-only
torch.load("m.pt", weights_only=True)

# ruleid: numpy-load-allow-pickle
np.load("a.npy", allow_pickle=True)
# ok: numpy-load-allow-pickle
np.load("a.npy")

# ruleid: keras-load-model-unsafe
keras.models.load_model("m.keras", safe_mode=False)

# ruleid: hf-trust-remote-code, hf-from-pretrained-unpinned
AutoModel.from_pretrained("org/model", trust_remote_code=True)

# ruleid: torch-hub-load
torch.hub.load("pytorch/vision", "resnet18")

# ruleid: hf-from-pretrained-unpinned
AutoModel.from_pretrained("org/model")
# ok: hf-from-pretrained-unpinned
AutoModel.from_pretrained("org/model", revision="abc123")
# ok: hf-from-pretrained-unpinned
AutoModel.from_pretrained("./local_dir")

# ruleid: hf-download-unpinned
hf_hub_download("org/model", "config.json")
# ok: hf-download-unpinned
hf_hub_download("org/model", "config.json", revision="abc123")
# ruleid: hf-download-unpinned
datasets.load_dataset("org/data")


def agent(client):
    resp = client.messages.create(model="m", messages=[])
    cmd = resp.content[0].text
    # ruleid: llm-output-to-code-execution
    subprocess.run(cmd, shell=True)
    # ruleid: llm-output-to-code-execution
    os.system(cmd)
    safe = validate_tool_call(cmd)
    # ok: llm-output-to-code-execution
    subprocess.run(safe)
