import torch
import subprocess


def get_vram_usage():
    used_megabytes = subprocess.getoutput(
        "echo `nvidia-smi --query-gpu=memory.used --format=csv|grep -v memory|awk '{print $1}')Mb usados na GPU`")
    if torch.cuda.is_available():
        return float(used_megabytes)
    else:
        return 0
