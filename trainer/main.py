"""
Multi-GPU training (single node, 4 training + 4 inference)

Run from repo root via `modal run -m trainer`

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml verifiers/examples/math_train.py
"""
import os
import subprocess

import modal

app = modal.App("verifiers-math-training", secrets=[
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("wandb-secret"),
])

cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .pip_install("uv")
    .run_commands("uv pip install --system wheel")
    .run_commands("apt-get -y update && apt-get -y install git")
    # .add_local_dir("../verifiers", "/verifiers_src", copy=True)
    # .run_commands("uv pip install --system /verifiers_src", gpu="H100")
    .run_commands("uv pip install --system git+https://github.com/jvmncs/verifiers.git", gpu="H100")
    .run_commands("uv pip install --system --no-build-isolation flash-attn", gpu="H100")
)
MINUTES = 60
HOURS = 60 * MINUTES
@app.function(
    image=image,
    gpu="H100:8",
    scaledown_window=None,
    timeout=2 * HOURS
)
def main():
    procs = []
    jobs = [
        (
            [
                "python", "-m", "verifiers.inference.vllm_serve",
                "--model", "Qwen/Qwen2.5-7B-Instruct",
                "--tensor_parallel_size", "4",
                "--max_model_len", "8192",
                "--dtype", "bfloat16",
                "--gpu_memory_utilization", "0.9",
                "--enable_prefix_caching", "True",
                "--host", "0.0.0.0",
                "--port", "8000",
            ],
            {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        ),
        (
            [
                "accelerate", "launch", "--config-file", "trainer/zero3.yaml", "trainer/math_train.py"
            ],
            {"CUDA_VISIBLE_DEVICES": "4,5,6,7"},
        ),
    ]
    for cmd, env_overrides in jobs:
        env = os.environ.copy()
        env.update(env_overrides)
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)

    try:
        # Training process
        procs[1].wait()

        if procs[0].poll() is None:  # If still running
            procs[0].terminate()
            procs[0].wait()
    except KeyboardInterrupt:
        # Gracefully terminate processes on interrupt
        for p in procs:
            if p.poll() is None:  # If still running
                p.terminate()
                p.wait()
        raise

    # Return based on training process success
    return procs[1].returncode == 0
