# UniRig RunPod Serverless Worker

Обёртка над [VAST-AI-Research/UniRig](https://github.com/VAST-AI-Research/UniRig)
(SIGGRAPH'25) под RunPod Serverless: на входе — 3D-меш (`.obj`/`.fbx`/`.glb`/`.vrm`),
на выходе — тот же меш, но со скелетом и (опционально) весами скиннинга.

## Как это устроено внутри

UniRig — это не одна модель, а пайплайн из трёх официальных bash-скриптов
(`launch/inference/*.sh`), которые дергает `handler.py`:

1. **`generate_skeleton.sh`** — автогрегрессивный трансформер предсказывает
   иерархию скелета (кости + их вложенность) по геометрии меша → `.fbx`.
2. **`generate_skin.sh`** — Bone-Point Cross Attention предсказывает веса
   скиннинга по вершинам, используя скелет из шага 1 → `.fbx`.
3. **`merge.sh`** — сливает предсказанный скелет/скин обратно с исходной
   геометрией/текстурами в финальный `.glb`/`.fbx`.

Мы намеренно НЕ переписываем внутреннюю логику (hydra-конфиги, tokenizer,
шейп-энкодер) на прямые вызовы Python-функций — она сложная, завязана на
конфиги `configs/task/*.yaml` и может измениться в апстриме. Вместо этого
`handler.py` дергает официальные скрипты через `subprocess`, ровно как
показано в README самого UniRig. Это осознанный компромисс: чуть медленнее
(лишний процесс на каждый шаг), зато не расклеится при `git pull` апдейта
репозитория.

## Формат входа/выхода (`handler.py`)

```json
{
  "input": {
    "mesh_base64": "...",           // либо mesh_url — одно из двух обязательно
    "mesh_url": "https://...",
    "input_format": "glb",          // obj | fbx | glb | vrm (нужен, если не виден в URL)
    "seed": 42,                     // опционально — вариативность скелета
    "skin": true,                   // false -> вернётся только скелет без скиннинга
    "output_format": "glb",         // glb | fbx
    "return_skeleton": false        // вернуть также промежуточный skeleton.fbx
  }
}
```

Ответ:

```json
{
  "rigged_mesh_base64": "...",   // или rigged_mesh_url, если настроен BUCKET_ENDPOINT_URL
  "format": "glb",
  "skinned": true
}
```

## Деплой

1. Запушь этот репозиторий (Dockerfile, handler.py, workflow) на GitHub —
   Actions сам соберёт образ и запушит в `ghcr.io/<owner>/<repo>:latest`.
2. В RunPod → Serverless → New Endpoint → Custom Image → указать
   `ghcr.io/<owner>/<repo>:latest` (если пакет приватный — добавить registry
   credentials в настройках эндпоинта).
3. GPU: минимум **8GB VRAM** (по факту README) — комфортно берётся
   16GB+ (RTX 4090 / A5000 / L4), т.к. skin-стадия прожорливее skeleton-стадии.
4. Опционально: `BUCKET_ENDPOINT_URL` + сопутствующие RunPod env-переменные
   для S3-совместимого бакета — тогда в ответе прилетит `*_url`, а не
   base64 (полезно для больших `.fbx` с текстурами).

## Известные подводные камни (уже учтены в Dockerfile, но держи в уме)

- **Python обязан быть 3.11.** `bpy==4.2` (headless Blender) на PyPI
  собран только под `cp310`/`cp311` linux — на 3.12/3.10 без танцев
  с бубном не встанет. Поэтому в образе явно ставится Python 3.11 через
  deadsnakes, а не системный из Ubuntu 22.04 (там 3.10, могло бы даже
  завестись, но 3.11 безопаснее — так тестировал апстрим).
- **Порядок пакетов важен.** `numpy==1.26.4` ставится строго последним —
  если поставить раньше, transformers/lightning подтянут более свежий
  numpy как транзитивную зависимость и молча его переустановят, а
  `spconv`/`open3d` на новом numpy иногда падают по ABI на рантайме,
  а не на установке (самый неприятный тип бага).
- **`spconv` и `torch_scatter`/`torch_cluster` жёстко привязаны к**
  **связке torch+CUDA.** Меняешь версию torch — обязательно меняй
  `spconv-cu1XX` и URL `data.pyg.org/whl/torch-X.Y.Z+cuXXX.html` синхронно,
  иначе поймаешь `ImportError: undefined symbol` при первом реальном
  инференсе (на установке ошибки не будет).
- **`flash_attn`** — самый долгий и самый капризный шаг сборки. В
  Dockerfile сначала пробуем готовый wheel с официальных релизов
  Dao-AILab (секунды), и только если под твою связку torch/cuda/python
  такого wheel нет — падаем в сборку из исходников
  (`--no-build-isolation`, `MAX_JOBS=4`, чтобы не улететь по RAM на
  раннере GitHub Actions). Если увидишь на билде `OOM` при линковке —
  снижай `MAX_JOBS` до 1-2 в Dockerfile.
- **`pyrender` внутри headless-контейнера** иногда не находит дисплей
  для offscreen-рендеринга вокселей при обучении/некоторых конфигурациях
  скиннинга — это прямо описано в апстрим README как известная проблема,
  с рекомендацией переключить `voxel_skin/backend` на `open3d`. Dockerfile
  проактивно патчит это `sed`'ом по всем `configs/**/*.yaml` (безопасно —
  если строки нет, `sed` просто ничего не делает).
- **GitHub Actions free runner:** ~14GB диска, ~7GB RAM. Итоговый образ
  (CUDA devel + torch + flash_attn + предзагруженные веса ~5.5GB) идёт
  впритык — в workflow уже добавлен шаг очистки диска перед сборкой.
  Если всё равно не хватает места — переходи на self-hosted runner
  или собирай образ прямо на своём RunPod-сервере через `docker build`.
- **VRM поддержка опциональна.** Blender VRM addon ставится best-effort
  (`|| echo ... non-fatal`) — если билд идёт без addon, `.vrm` на входе/
  выходе просто не будет обрабатываться, а `.obj/.fbx/.glb` продолжат
  работать как обычно.

## Локальный тест (без RunPod, прямо в контейнере)

```bash
docker build -t unirig-worker .
docker run --gpus all -it unirig-worker \
    python3 -u handler.py --test_input "$(cat test_input.json)"
```

`test_input.json` в этом репозитории указывает на пример `giraffe.glb`
из самого репозитория UniRig — удобно для быстрой проверки, что вся
цепочка (skeleton → skin → merge) вообще доезжает до конца, прежде чем
гонять на своих реальных ассетах.
