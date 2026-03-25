# ReconVision

ReconVision - сервис обработки актов сверки.

Сервис принимает PDF в base64, строит каноническое техническое представление документа, извлекает структурированные данные и по сохранённому состоянию процесса может сформировать заполненный PDF.

Проект разделён на два уровня:

- `src/vision_core` - независимое техническое ядро OCR, layout-анализа и сборки канонического `Document`
- `src/app` - сервисный слой FastAPI поверх `vision_core`

## Что делает сервис

Текущий сценарий работы:

1. Клиент отправляет PDF через `send_reconciliation_act`
2. Сервис декодирует base64 и создаёт `ProcessState`
3. `vision_core` строит канонический `Document`
4. Прикладной слой сохраняет исходный PDF, `Document`, статус и извлечённые данные процесса
5. Клиент опрашивает `process_status`
6. После завершения обработки клиент может вызвать `fill_reconciliation_act`
7. PDF filler записывает данные обратно в исходный PDF по координатам строк, найденных в `Document`

## Архитектура

### `src/app`

Сервисный слой построен как явная слоистая архитектура:

- `api` - HTTP transport, схемы запросов и ответов, маршруты FastAPI
- `application` - use case-логика и порты
- `domain` - бизнес-сущности сервиса: процесс, данные акта сверки, бухгалтерские записи, период
- `infrastructure` - реализации портов: persistence, vision adapter, PDF filler, config, logging
- `bootstrap` - composition root и wiring зависимостей

Зависимости направлены так:

`api -> application -> domain`

`infrastructure` реализует порты `application`, но не протаскивает transport-логику внутрь домена.

### `src/vision_core`

`vision_core` не зависит от `app` и отвечает только за технический пайплайн документа:

- загрузка PDF и рендеринг страниц
- preprocessing изображений
- OCR и layout analysis
- извлечение таблиц и абзацев
- сборка канонического `Document`
- геометрические координаты, используемые fill-сценарием

Центральный технический контракт - `vision_core.entities.document.Document`.

## Канонические модели

Ключевые внутренние модели:

- `ProcessState` - состояние процесса обработки, включая исходный PDF, `Document`, статус и извлечённые данные
- `ReconciliationData` - структурированный результат обработки акта сверки
- `LedgerEntry` - одна запись дебета или кредита
- `Document` - каноническое техническое представление документа

Важно:

- `Document` - технический контракт между `vision_core` и `app`
- transport-модели API не должны утекать в `domain`
- смена persistence или PDF filler не должна требовать переписывания use case-логики

## Структура репозитория

Актуальная структура верхнего уровня:

- `src/app` - FastAPI сервис
- `src/vision_core` - OCR и document pipeline
- `tests` - unit и integration тесты
- `models` - OCR-модели PaddleOCR
- `examples` - тестовые и демонстрационные PDF
- `notebooks` - исследовательские ноутбуки
- `static` - локальная swagger/redoc статика

## Внешний API

Базовый префикс API: `/api/v1`

### `POST /send_reconciliation_act`

Request:

```json
{
    "document": "<base64 pdf>"
}
```

Response `201`:

```json
{
    "process_id": "string"
}
```

Response `400`:

```json
{
    "status": -2,
    "message": "Не удалось декодировать PDF из base64"
}
```

### `POST /process_status`

Request:

```json
{
    "process_id": "string"
}
```

Response `201` while processing:

```json
{
    "status": 0,
    "message": "wait"
}
```

Response `404`:

```json
{
    "status": -1,
    "message": "not found"
}
```

Response `500`:

```json
{
    "status": -2,
    "message": "<error description>"
}
```

Response `200`:

```json
{
    "process_id": "string",
    "status": 0,
    "message": "string",
    "seller": "string",
    "buyer": "string",
    "period": {
        "start": "string",
        "end": "string"
    },
    "debit": [
        {
            "row_id": {
                "id_row": "string",
                "id_table": "string"
            },
            "record": "string",
            "value": 0,
            "date": "string"
        }
    ],
    "credit": [
        {
            "row_id": {
                "id_row": "string",
                "id_table": "string"
            },
            "record": "string",
            "value": 0,
            "date": "string"
        }
    ]
}
```

Примечание: публичный контракт использует одинаковое поле `status = 0` и для `wait`, и для успешного ответа `200`. Различать состояния нужно по HTTP-коду и составу payload.

### `POST /fill_reconciliation_act`

Request:

```json
{
    "process_id": "string",
    "comments": "string",
    "debit": [
        {
            "row_id": {
                "id_row": "string",
                "id_table": "string"
            },
            "record": "string",
            "value": 0,
            "date": "string"
        }
    ],
    "credit": [
        {
            "row_id": {
                "id_row": "string",
                "id_table": "string"
            },
            "record": "string",
            "value": 0,
            "date": "string"
        }
    ]
}
```

Response `200`:

```json
{
    "document": "<base64 pdf>"
}
```

Ошибки:

- `201` - процесс ещё не готов
- `404` - процесс не найден
- `500` - ошибка заполнения

## Текущее состояние реализации

Что уже реализовано:

- `src-layout` и слоистая архитектура сервиса
- каноническая сущность `Document`
- `DocumentBuildPipeline`
- FastAPI API поверх use cases
- хранение состояния процесса через DiskCache
- контрактные HTTP-тесты
- use case-тесты
- fill-сценарий поверх координат `Document`

Что пока упрощено:

- semantic extraction пока реализован через временный adapter `StubStructuredDataExtractor`
- fill-сценарий распределяет данные по ячейкам строки эвристически слева направо
- комментарий вставляется на первую страницу, а не в отдельную семантическую область шаблона

## Требования для разработки

- Python `3.11.9`
- `uv`

## Установка

```sh
make install PYTHON=python3.11
```

Если Python `3.11.9` установлен через `pyenv`, можно передать полный путь:

```sh
make install PYTHON="$HOME/.pyenv/versions/3.11.9/bin/python"
```

`Makefile` проверяет точную версию Python, создаёт `.venv` и ставит зависимости через `uv`.

## Локальный запуск

Надёжная команда для локального запуска сервиса:

```sh
source .venv/bin/activate
PYTHONPATH=src uv run --python .venv/bin/python uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Документация после запуска доступна по адресам:

- `http://localhost:8000/api/v1/docs`
- `http://localhost:8000/api/v1/redoc`

## Тестирование

Полный прогон:

```sh
make test
```

Полезные точечные команды:

```sh
source .venv/bin/activate
pytest tests/integration/test_api_contract.py -q
pytest tests/unit/test_use_cases_unit.py -q
pytest tests/unit/test_pdf_filler_unit.py -q
pytest tests/vision_core/entities/test_document.py tests/vision_core/pipelines/test_build_document.py -q
```

## Docker

Сборка и запуск контейнеров:

```sh
make deploy
```

Остановка:

```sh
make down
```
