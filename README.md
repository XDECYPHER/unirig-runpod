# UniRig — RunPod Serverless Deployer

Оборачивает [UniRig](https://github.com/VAST-AI-Research/UniRig) (skeleton + skinning prediction)
в Docker-образ и RunPod Serverless handler, чтобы дёргать его как обычный API из своего Python-кода.

## Структура репозитория

```
.
├── Dockerfile                        # образ: CUDA + UniRig + веса модели + handler
├── handler.py                        # RunPod serverless handler (входная точка контейнера)
├── client_example.py                 # пример вызова endpoint'а из своего кода
├── test_input.json                   # пример payload для локального теста
└── .github/workflows/docker-build.yml # авто-сборка и пуш образа в GHCR при пуше в main
```

## Шаг 1. Репозиторий на GitHub

1. Создай новый **приватный или публичный** репозиторий на GitHub.
2. Залей туда все файлы из этой папки.
3. Ничего дополнительно настраивать не нужно — `GITHUB_TOKEN` для пуша в GHCR
   выдаётся Actions автоматически, свой токен создавать не надо.
4. Сделай `git push` в `main` — запустится workflow `docker-build.yml` и соберёт образ:

   ```
   ghcr.io/<твой-github-username>/unirig-runpod:latest
   ```

   Собирается он долго (10–20+ минут) — там CUDA-toolchain, spconv, flash-attn и т.д.
   Смотри прогресс во вкладке **Actions** репозитория.

5. Сделай образ публичным (иначе RunPod не сможет его вытянуть без доп. настройки):
   зайди в GitHub → твой профиль → **Packages** → `unirig-runpod` → **Package settings** →
   **Change visibility** → Public. Либо добавь registry credentials в RunPod (см. шаг 2).

## Шаг 2. Endpoint на RunPod

1. RunPod Console → **Serverless** → **New Endpoint**.
2. **Container Image**: `ghcr.io/<твой-username>/unirig-runpod:latest`
   (если образ приватный — добавь Container Registry Credentials в настройках RunPod).
3. **GPU**: минимум 8GB VRAM (в UniRig это официальный минимум), но безопаснее
   брать 16GB (RTX A4000 / RTX 4000 Ada и т.п.) — с запасом под skinning-этап.
4. **Container Disk**: 20–30GB (веса модели + CUDA-либы).
5. Остальное (min/max workers, idle timeout) — по вкусу, для старта можно
   min workers = 0, max = 1–2.
6. Создай endpoint, скопируй его **Endpoint ID** и свой **API Key**
   (RunPod → Settings → API Keys).

## Шаг 3. Вызов из своего Python-кода

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
python client_example.py
```

`client_example.py` — рабочий шаблон: берёшь модель, кодируешь в base64 (или отдаёшь
`model_url`, если модель уже где-то лежит), шлёшь на `/run`, поллишь `/status/<id>`,
получаешь результат обратно в base64.

### Формат запроса к handler'у

```json
{
  "input": {
    "task": "full",          // "skeleton" | "skin" | "full"
    "input_ext": "glb",      // расширение входного файла
    "model_base64": "...",   // ИЛИ "model_url": "https://..."
    "seed": 42                // опционально, только для этапа skeleton
  }
}
```

- `skeleton` — только предсказание скелета, вернёт `.fbx` со скелетом.
- `skin` — только предсказание весов скиннинга (нужен уже готовый skeleton `.fbx` на входе).
- `full` — skeleton → skin → merge, на выходе полностью зарикованный `.glb`.

## На что обратить внимание

- **Версии CUDA/torch в Dockerfile — ориентир, не догма.** Я взял `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-...`
  как базовый образ и подобрал под него `spconv-cu124` и `torch_scatter/torch_cluster` с сайта PyG.
  Перед первым билдом стоит проверить, что тег базового образа ещё существует на
  [Docker Hub](https://hub.docker.com/r/runpod/pytorch/tags) — RunPod время от времени убирает старые теги.
- **flash-attn** — самое капризное место при сборке (см. предупреждение прямо в README UniRig).
  Если билд на этом шаге падает — иди в [репозиторий flash-attention](https://github.com/Dao-AILab/flash-attention),
  найди wheel под свою связку python/torch/cuda и пропиши версию явно в Dockerfile.
- Веса модели (`VAST-AI/UniRig` с Hugging Face) скачиваются **во время сборки образа**,
  а не при каждом холодном старте — это заметно ускоряет первый запрос к serverless-воркеру.
- Для реально долгих генераций (`full` на тяжёлых мешах) используй `/run` + поллинг статуса,
  а не `/runsync` — синхронный вызов может упереться в таймаут RunPod.
