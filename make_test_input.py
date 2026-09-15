"""
Генерирует test_input.json из твоей реальной локальной 3D-модели.

Зачем: старый test_input.json ссылался на
https://raw.githubusercontent.com/VAST-AI-Research/UniRig/main/examples/giraffe.glb
— а raw.githubusercontent.com отдаёт для Git LFS файлов не сам .glb,
а текстовый LFS pointer (~116 байт), поэтому Blender падал на парсинге glTF.

Использование:
    python make_test_input.py /path/to/my_model.glb --task full --seed 42

Результат: test_input.json с твоей моделью в base64 — можно кормить
и в handler.py напрямую, и в RunPod (через /run), и в local_test_api.py.
"""

import argparse
import base64
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path", help="Путь к своему .glb/.fbx/.obj файлу")
    p.add_argument("--task", default="full", choices=["skeleton", "skin", "full"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="test_input.json")
    args = p.parse_args()

    ext = args.model_path.rsplit(".", 1)[-1].lower()
    with open(args.model_path, "rb") as f:
        model_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "task": args.task,
            "input_ext": ext,
            "model_base64": model_b64,
            "seed": args.seed,
        }
    }

    with open(args.out, "w") as f:
        json.dump(payload, f)

    print(f"Готово: {args.out} ({len(model_b64)} байт base64, task={args.task}, ext={ext})")


if __name__ == "__main__":
    main()
