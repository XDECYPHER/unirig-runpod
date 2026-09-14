"""
RunPod Serverless handler для UniRig (3D mesh -> rigged mesh: skeleton + skinning).

Официальный репозиторий: https://github.com/VAST-AI-Research/UniRig
Веса: https://huggingface.co/VAST-AI/UniRig (скачаны заранее на этапе Docker build)

Полный пайплайн UniRig состоит из 3 шагов, каждый — отдельный bash-скрипт
в launch/inference/:
    1. generate_skeleton.sh  -> предсказывает скелет (.fbx)
    2. generate_skin.sh      -> предсказывает веса скиннинга по скелету (.fbx)
    3. merge.sh               -> сливает результат с исходным мешем (.glb/.fbx)

Мы не переписываем внутреннюю логику (hydra-конфиги, tokenizer и т.д.) —
она сложная и может измениться в апстриме, поэтому дергаем её ровно так,
как это делает README, через subprocess. Так при обновлении репозитория
(`git pull` внутри образа) наш handler не расклеится.

Ожидаемый вход (event["input"]):
{
    "mesh_base64": "<base64 файла меша>",   # либо
    "mesh_url": "https://...",               # одно из двух обязательно
    "input_format": "glb",                   # опционально, если не удаётся
                                              # определить по URL/имени файла.
                                              # Поддерживается: obj, fbx, glb, vrm
    "seed": 42,                              # опционально, вариативность скелета
    "skin": true,                            # опционально (default True) —
                                              # если False, вернётся только
                                              # скелет без весов скиннинга
    "output_format": "glb",                  # glb | fbx, default "glb"
    "return_skeleton": false                 # опционально, вернуть также
                                              # промежуточный файл скелета
}

Выход:
{
    "rigged_mesh_base64": "...",   # либо rigged_mesh_url, если настроен bucket
    "skeleton_fbx_base64": "...",  # только если return_skeleton=true
    "format": "glb"
}
"""

import os
import sys
import base64
import shutil
import traceback
import tempfile
import subprocess

import runpod
import requests

REPO_DIR = "/workspace/UniRig"
LAUNCH_DIR = os.path.join(REPO_DIR, "launch", "inference")

SUPPORTED_INPUT_FORMATS = {"obj", "fbx", "glb", "vrm"}
SUPPORTED_OUTPUT_FORMATS = {"glb", "fbx"}


def _guess_ext(job_input: dict) -> str:
    """Пытаемся понять формат входного меша: явный параметр -> URL -> дефолт."""
    fmt = (job_input.get("input_format") or "").strip(".").lower()
    if fmt in SUPPORTED_INPUT_FORMATS:
        return fmt

    url = job_input.get("mesh_url", "")
    for candidate in SUPPORTED_INPUT_FORMATS:
        if url.lower().split("?")[0].endswith("." + candidate):
            return candidate

    raise ValueError(
        "Не удалось определить формат входного меша. "
        f"Укажи 'input_format' явно (один из {sorted(SUPPORTED_INPUT_FORMATS)})."
    )


def _load_mesh(job_input: dict, dst_path: str) -> None:
    """Сохраняем входной меш на диск из base64 либо по URL."""
    if job_input.get("mesh_base64"):
        raw = base64.b64decode(job_input["mesh_base64"])
        with open(dst_path, "wb") as f:
            f.write(raw)
        return

    if job_input.get("mesh_url"):
        resp = requests.get(job_input["mesh_url"], timeout=60)
        resp.raise_for_status()
        with open(dst_path, "wb") as f:
            f.write(resp.content)
        return

    raise ValueError("Нужно передать 'mesh_base64' или 'mesh_url' во входных данных.")


def _run(cmd: list, cwd: str = REPO_DIR) -> None:
    """Запускаем шаг пайплайна и подробно логируем вывод при ошибке."""
    print(f"[UniRig] $ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Шаг пайплайна упал (exit={proc.returncode}): {' '.join(cmd)}\n"
            f"--- вывод ---\n{proc.stdout[-4000:]}"
        )


def _upload_or_encode(file_path: str, key_prefix: str) -> dict:
    """Заливка в bucket (если настроен через RunPod env), либо base64 в ответе."""
    try:
        from runpod.serverless.utils import rp_upload

        if os.environ.get("BUCKET_ENDPOINT_URL"):
            url = rp_upload.upload_file_to_bucket(
                file_name=os.path.basename(file_path),
                file_location=file_path,
            )
            return {f"{key_prefix}_url": url}
    except Exception as e:
        print(f"[UniRig] Не удалось загрузить в bucket, fallback на base64: {e}")

    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return {f"{key_prefix}_base64": encoded}


def handler(event):
    tmp_dir = tempfile.mkdtemp(prefix="unirig_")
    try:
        job_input = event.get("input", {})

        input_ext = _guess_ext(job_input)
        output_format = (job_input.get("output_format") or "glb").lower()
        if output_format not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"output_format должен быть одним из {SUPPORTED_OUTPUT_FORMATS}")

        do_skin = bool(job_input.get("skin", True))
        return_skeleton = bool(job_input.get("return_skeleton", False))
        seed = job_input.get("seed")

        # --- пути к промежуточным/итоговым файлам ---
        src_mesh = os.path.join(tmp_dir, f"input.{input_ext}")
        skeleton_fbx = os.path.join(tmp_dir, "skeleton.fbx")
        skin_fbx = os.path.join(tmp_dir, "skin.fbx")
        rigged_out = os.path.join(tmp_dir, f"rigged.{output_format}")

        _load_mesh(job_input, src_mesh)

        # --- 1) Skeleton prediction ---
        skeleton_cmd = [
            "bash", os.path.join(LAUNCH_DIR, "generate_skeleton.sh"),
            "--input", src_mesh,
            "--output", skeleton_fbx,
        ]
        if seed is not None:
            skeleton_cmd += ["--seed", str(int(seed))]
        _run(skeleton_cmd)

        result = {}
        merge_source = skeleton_fbx

        # --- 2) Skinning weight prediction (опционально) ---
        if do_skin:
            skin_cmd = [
                "bash", os.path.join(LAUNCH_DIR, "generate_skin.sh"),
                "--input", skeleton_fbx,
                "--output", skin_fbx,
            ]
            _run(skin_cmd)
            merge_source = skin_fbx

        # --- 3) Merge skeleton/skin обратно в исходный меш ---
        merge_cmd = [
            "bash", os.path.join(LAUNCH_DIR, "merge.sh"),
            "--source", merge_source,
            "--target", src_mesh,
            "--output", rigged_out,
        ]
        _run(merge_cmd)

        result.update(_upload_or_encode(rigged_out, "rigged_mesh"))
        result["format"] = output_format
        result["skinned"] = do_skin

        if return_skeleton:
            result.update(_upload_or_encode(skeleton_fbx, "skeleton_fbx"))

        return result

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
