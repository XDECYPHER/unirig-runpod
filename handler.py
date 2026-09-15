"""
RunPod serverless handler for UniRig.

Accepts a 3D model (base64 or URL), runs one of UniRig's inference stages,
and returns the result as base64.

Expected job input:
{
  "input": {
    "task": "skeleton" | "skin" | "full",   # default "skeleton"
    "input_ext": "glb",                     # extension of the uploaded model, default "glb"
    "model_base64": "<...>",                # OR
    "model_url": "https://.../model.glb",
    "seed": 42                              # optional, only used for skeleton stage
  }
}

Response:
{
  "output_base64": "<...>",
  "output_ext": "fbx" | "glb",
  "task": "skeleton"
}
"""

import base64
import os
import subprocess
import tempfile
import traceback

import requests
import runpod

REPO_DIR = "/workspace/UniRig"
os.chdir(REPO_DIR)


def _save_input_file(job_input: dict, workdir: str) -> str:
    """Save the incoming 3D model (base64 or URL) to disk and return its path."""
    ext = job_input.get("input_ext", "glb").lstrip(".")
    in_path = os.path.join(workdir, f"input.{ext}")

    if job_input.get("model_base64"):
        with open(in_path, "wb") as f:
            f.write(base64.b64decode(job_input["model_base64"]))
    elif job_input.get("model_url"):
        r = requests.get(job_input["model_url"], timeout=120)
        r.raise_for_status()
        with open(in_path, "wb") as f:
            f.write(r.content)
    else:
        raise ValueError("Provide either 'model_base64' or 'model_url' in input.")

    return in_path


def _run(cmd: list) -> None:
    result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed: "
            + " ".join(cmd)
            + f"\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def handler(job):
    job_input = job.get("input", {}) or {}
    task = job_input.get("task", "skeleton")
    seed = str(job_input.get("seed", 42))

    workdir = tempfile.mkdtemp(prefix="unirig_")

    try:
        in_path = _save_input_file(job_input, workdir)

        if task == "skeleton":
            out_path = os.path.join(workdir, "skeleton.fbx")
            out_ext = "fbx"
            _run([
                "bash", "launch/inference/generate_skeleton.sh",
                "--input", in_path, "--output", out_path, "--seed", seed,
            ])

        elif task == "skin":
            out_path = os.path.join(workdir, "skin.fbx")
            out_ext = "fbx"
            _run([
                "bash", "launch/inference/generate_skin.sh",
                "--input", in_path, "--output", out_path,
            ])

        elif task == "full":
            # skeleton -> skin -> merge back onto the original mesh
            skel_path = os.path.join(workdir, "skeleton.fbx")
            skin_path = os.path.join(workdir, "skin.fbx")
            out_path = os.path.join(workdir, "rigged.glb")
            out_ext = "glb"

            _run([
                "bash", "launch/inference/generate_skeleton.sh",
                "--input", in_path, "--output", skel_path, "--seed", seed,
            ])
            _run([
                "bash", "launch/inference/generate_skin.sh",
                "--input", skel_path, "--output", skin_path,
            ])
            _run([
                "bash", "launch/inference/merge.sh",
                "--source", skin_path, "--target", in_path, "--output", out_path,
            ])

        else:
            raise ValueError(f"Unknown task '{task}'. Use 'skeleton', 'skin', or 'full'.")

        with open(out_path, "rb") as f:
            out_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {"output_base64": out_b64, "output_ext": out_ext, "task": task}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
