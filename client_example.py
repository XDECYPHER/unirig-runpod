"""
Minimal example of calling the deployed UniRig RunPod serverless endpoint
from your own Python code (e.g. where you already pick which 3D model to rig).

Set these two env vars first:
  RUNPOD_API_KEY   - from RunPod > Settings > API Keys
  RUNPOD_ENDPOINT_ID - from your Serverless endpoint's dashboard page
"""

import base64
import os
import time

import requests

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def rig_model(model_path: str, task: str = "full", seed: int = 42) -> bytes:
    ext = model_path.rsplit(".", 1)[-1]
    with open(model_path, "rb") as f:
        model_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "task": task,
            "input_ext": ext,
            "model_base64": model_b64,
            "seed": seed,
        }
    }

    # /run = async job; poll /status until it finishes.
    # (use /runsync instead if you want a single blocking call and your
    # job reliably finishes within RunPod's sync timeout)
    resp = requests.post(f"{BASE_URL}/run", headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    job_id = resp.json()["id"]

    while True:
        status_resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=30)
        status_resp.raise_for_status()
        status = status_resp.json()

        if status["status"] == "COMPLETED":
            output = status["output"]
            if "error" in output:
                raise RuntimeError(f"Worker error: {output['error']}\n{output.get('trace', '')}")
            return base64.b64decode(output["output_base64"]), output["output_ext"]

        if status["status"] == "FAILED":
            raise RuntimeError(f"Job failed: {status}")

        time.sleep(3)


if __name__ == "__main__":
    data, ext = rig_model("examples/giraffe.glb", task="full")
    out_name = f"rigged_result.{ext}"
    with open(out_name, "wb") as f:
        f.write(data)
    print(f"Saved {out_name}")
