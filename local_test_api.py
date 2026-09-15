"""
Локальный тестовый сервер для UniRig-хендлера — без RunPod и без сети.

Зачем: чтобы не гонять test_input.json с чужим giraffe.glb (который у GitHub
всё равно отдаётся как Git LFS pointer, а не как реальный файл), а просто
залить СВОЙ .glb/.fbx/.obj и сразу увидеть, работает пайплайн или нет.

Запуск (внутри контейнера/GPU-машины, там же, где стоит UniRig):
    pip install fastapi uvicorn python-multipart
    python local_test_api.py
    # сервер поднимется на http://localhost:8002

Тест через curl:
    curl -X POST http://localhost:8002/rig \
      -F "file=@/path/to/my_model.glb" \
      -F "task=full" \
      -F "seed=42" \
      -o rigged_result.glb

Или открой в браузере http://localhost:8002/docs — там будет автосгенерённая
форма для загрузки файла (Swagger UI), можно тестировать мышкой без curl.
"""

import base64

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

# handler.py должен лежать рядом (или быть на PYTHONPATH) — это тот же файл,
# что запускается в контейнере через runpod.serverless.start(...).
# Импортируем именно функцию handler(), а не сам модуль как entrypoint —
# runpod.serverless.start() внутри handler.py не мешает импорту, т.к. он
# вызывается только под `if __name__ == "__main__"`... но в текущем handler.py
# он вызывается на верхнем уровне модуля, поэтому см. примечание ниже.
from handler import handler as unirig_handler

app = FastAPI(title="UniRig local test API")


@app.post("/rig")
async def rig(
    file: UploadFile = File(..., description="Твоя 3D-модель (.glb/.fbx/.obj)"),
    task: str = Form("full", description="skeleton | skin | full"),
    seed: int = Form(42),
):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    raw = await file.read()
    model_b64 = base64.b64encode(raw).decode("utf-8")

    job = {
        "input": {
            "task": task,
            "input_ext": ext,
            "model_base64": model_b64,
            "seed": seed,
        }
    }

    result = unirig_handler(job)

    if "error" in result:
        # Отдаём полный traceback и содержимое workdir — как в проде на RunPod,
        # только сразу в ответе, без поллинга /status.
        return JSONResponse(status_code=500, content=result)

    out_bytes = base64.b64decode(result["output_base64"])
    out_ext = result["output_ext"]
    return Response(
        content=out_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="rigged_result.{out_ext}"'
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
