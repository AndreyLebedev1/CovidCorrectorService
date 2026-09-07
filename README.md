# CovidCorrectorService

Сервис инференса ResNet50 с посткорректором для выявления COVID-19 на рентгеновских изображениях грудной клетки.

## Структура

```text
.
├── bin/
│   ├── legacy_resnet50.pt
│   ├── full_hidden_preprocessor.joblib
│   └── full_hidden_fisher_corrector.joblib
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── covid_corrector/
├── main.py
└── model.py
```

Файлы модели и изображения хранятся в Git LFS.

## Подготовка репозитория и данных

Установите Git LFS и склонируйте репозиторий:

```powershell
git lfs install
git clone git@github.com:AndreyLebedev1/CovidCorrectorService.git
cd CovidCorrectorService
git lfs pull
```

После `git lfs pull` файлы в `bin/` должны быть полноценными бинарными файлами, а не текстовыми LFS-пойнтерами.

Для оценки датасет организован в формате `ImageFolder`:

```text
data/
├── train/
│   ├── COVID19/
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   └── TURBERCULOSIS/
├── val/
└── test/
```

Изображения должны находиться внутри папок соответствующих классов. Поддерживаются форматы `.jpg`, `.jpeg` и `.png`.

Для обычного инференса весь датасет не требуется: достаточно иметь локальный файл изображения и три артефакта в `bin/`.

## Установка зависимостей через uv

Требуется Python 3.11. Установите `uv`, если он ещё не установлен:

```powershell
py -3.11 -m pip install uv
```

Затем из корня репозитория выполните:

```powershell
uv sync
```

Команда создаст виртуальное окружение `.venv` и установит зависимости из `pyproject.toml` согласно `uv.lock`.

Для CPU используется устройство `cpu` по умолчанию. При необходимости замените зависимости PyTorch на сборку, соответствующую своей CUDA-версии.

## Запуск сервиса

Из корня репозитория:

```powershell
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Интерактивная документация доступна по адресу:

```text
http://127.0.0.1:8000/docs
```

Если порт `8000` занят, используйте другой, например `8001`.

## Запуск в Docker

Docker-образ собирается на Python 3.11 и запускается от непривилегированного пользователя. Датасет `data/` намеренно не включается в образ: для инференса нужны только код и артефакты из `bin/`.

Перед сборкой убедитесь, что Git LFS-файлы скачаны:

```powershell
git lfs pull
docker build -t covidcorrectorservice:latest .
```

Запуск контейнера:

```powershell
docker run --rm -p 8000:8000 covidcorrectorservice:latest
```

После запуска API доступен по адресу `http://127.0.0.1:8000/docs`. Чтобы передавать контейнеру изображения с хоста, примонтируйте каталог:

```powershell
docker run --rm -p 8000:8000 -v "C:\path\to\images:/data:ro" covidcorrectorservice:latest
```

В запросе в этом случае указывается путь внутри контейнера, например `/data/image.jpg`, а не путь Windows-хоста.

## Инференс

Метод `POST /predict` принимает путь к изображению, доступный процессу сервиса:

```powershell
$body = @{
    path = "C:\path\to\project\CovidCorrectorService\data\test\COVID19\COVID19(460).jpg"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri http://127.0.0.1:8000/predict `
    -ContentType "application/json" `
    -Body $body
```

Пример ответа:

```json
{
  "legacy_prediction": "PNEUMONIA",
  "final_prediction": "COVID19",
  "correction_applied": true,
  "corrector_score": 0.42,
  "corrector_threshold": 0.0
}
```

Поля ответа:

- `legacy_prediction` — исходное решение ResNet50;
- `final_prediction` — решение после корректора;
- `correction_applied` — применено ли решение корректора;
- `corrector_score` — значение дискриминантной функции корректора;
- `corrector_threshold` — порог корректора.

Корректор использует скрытое пространство ResNet50, PCA-препроцессор и Fisher-дискриминант. Обучение в сервисе не выполняется: загружаются готовые артефакты из `bin/`.

## Типичные ошибки

- HTTP 404 — путь к изображению недоступен процессу сервиса;
- ошибка загрузки `.pt` или `.joblib` — выполните `git lfs pull`;
- `address already in use` — порт занят другим процессом, остановите его или выберите другой порт.
