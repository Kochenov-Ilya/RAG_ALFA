# Generator-first RAG pipeline для кейса «Альфа-Банк х МФТИ: RAG-ответы по базе знаний»

Репозиторий содержит готовый end-to-end ноутбук `alfa_rag_pipline.ipynb` для решения задачи построения RAG-системы по базе знаний Альфа-Банка. Пайплайн загружает данные, очищает корпус, строит чанки и индексы, выполняет hybrid retrieval, reranking, evidence grounding, генерацию ответа и сохраняет итоговый `submission.csv` в формате, ожидаемом платформой.

Основной принцип решения — **generator-first RAG**: retrieval и reranking используются для поиска релевантного evidence, grounding-блок собирает факты из найденных фрагментов, а финальный ответ формируется генератором на основе этого evidence. Экстрактивные ответы не используются как основной финальный режим, кроме аварийного fallback.

---

## 1. Цель проекта

Задача хакатона — построить RAG-пайплайн, который по пользовательскому вопросу:

1. Находит релевантные фрагменты в корпусе `websites.csv`.
2. Собирает grounded context из найденных фрагментов.
3. Генерирует связный, полезный и корректный ответ.
4. Сохраняет результат в CSV-файл с колонками:

```text
q_id, answer_new
```

Метрика лидерборда ориентирована на semantic recall ответа, поэтому пайплайн оптимизирован не на дословное извлечение фраз, а на смысловое покрытие релевантных условий, шагов, сроков, ограничений и исключений из базы знаний.

---

## 2. Основные особенности решения

В текущей версии реализованы следующие блоки:

### Preprocessing и chunking

* очистка HTML/разметки и технического мусора;
* нормализация банковских терминов;
* сохранение структуры текста и разделов;
* section-aware chunking;
* обработка длинных строк и абзацев;
* классификация страниц по типу источника;
* подготовка retrieval-friendly представления чанков.

### Query preprocessing

* нормализация пользовательского запроса;
* обработка коротких банковских запросов;
* query expansion для частых интентов;
* определение intent/profile запроса;
* query-level cache для одинаковых или почти одинаковых вопросов.

### Retrieval

* dense retrieval на Qwen embeddings;
* BM25 retrieval;
* hybrid retrieval через RRF;
* weighted query variants;
* source-type boosts;
* поддержка retrieval diagnostics.

### Reranking

* Qwen3 reranker;
* reranking top-N кандидатов из retrieval;
* комбинирование reranker score и hybrid retrieval score;
* подготовка компактного документа для reranker;
* сохранение diagnostic-признаков.

### Grounding и answerability

* выбор evidence chunks;
* parent/neighbor expansion;
* evidence compression;
* проверка answerability;
* отдельная обработка вопросов, где возможен общий ответ, но персональные данные недоступны;
* строгий no-answer режим: если информации действительно нет, ответ должен быть ровно `Нет ответа.`.

### Generation

* генерация ответа локальной open-weight LLM;
* decision-specific prompts;
* контроль длины ответа;
* проверка новых чисел и неподтверждённых фактов;
* post-processing;
* сохранение submission и diagnostics.

---

## 3. Используемые модели

В ноутбуке используются open-weight модели семейства Qwen:

```text
Embedding model: Qwen/Qwen3-Embedding-0.6B
Reranker model:  Qwen/Qwen3-Reranker-0.6B
Generator model: Qwen/Qwen2.5-7B-Instruct
```

Модели запускаются локально через Hugging Face Transformers. Для генератора используется 4-bit загрузка через `bitsandbytes`, если доступна CUDA.

---

## 4. Структура данных

Ожидается, что входные файлы находятся в директории:

```text
/content/drive/MyDrive/RAG
```

Но путь можно изменить в конфиге ноутбука:

```python
DATA_DIR = Path('/content/drive/MyDrive/RAG')
```

### Необходимые файлы

```text
questions.csv
websites.csv
sample_submission.csv
```

### `questions.csv`

Файл с пользовательскими вопросами.

Ожидаемые колонки:

```text
q_id, query
```

### `websites.csv`

Файл с распарсенными страницами сайта Альфа-Банка.

Ожидаемые колонки:

```text
web_id, url, title, text
```

Также поддерживаются альтернативные названия:

```text
website → url
web     → text
```

### `sample_submission.csv`

Используется только как ориентир по формату. Пайплайн не использует `sample_submission.csv` как источник знаний и не подгоняет ответы под него.

---

## 5. Выходные файлы

После полного запуска ноутбука сохраняются:

```text
submission_<RUN_NAME>.csv
diagnostics_<RUN_NAME>.csv
```

По умолчанию:

```text
submission_qwen_generator_first_v2_blocks_patch.csv
diagnostics_qwen_generator_first_v2_blocks_patch.csv
```

Если включён dev-режим через `DEV_N`, файлы сохраняются с суффиксом:

```text
_DEV<N>
```

Например:

```text
submission_qwen_generator_first_v2_blocks_patch_DEV100.csv
diagnostics_qwen_generator_first_v2_blocks_patch_DEV100.csv
```

---

## 6. Быстрый старт в Google Colab

1. Загрузить notebook `alfa_rag_pipline.ipynb` в Google Colab.

2. Подключить GPU:

```text
Runtime → Change runtime type → GPU
```

Рекомендуется GPU уровня L4/A100. На T4 пайплайн также может запускаться, но генерация будет медленнее.

3. Разместить файлы данных в Google Drive:

```text
/content/drive/MyDrive/RAG/questions.csv
/content/drive/MyDrive/RAG/websites.csv
/content/drive/MyDrive/RAG/sample_submission.csv
```

4. В первой конфигурационной ячейке проверить путь:

```python
DATA_DIR = Path('/content/drive/MyDrive/RAG')
```

5. Для тестового запуска установить:

```python
DEV_N = 100
```

6. Запустить все ячейки ноутбука.

7. После успешного dev-прогона запустить полный pipeline:

```python
DEV_N = None
FORCE_REBUILD_CHUNKS = True
FORCE_REBUILD_INDEX = True
FORCE_REGENERATE = True
```

8. Итоговый файл для отправки будет сохранён в `DATA_DIR`.

---

## 7. Конфигурация

Основные параметры находятся в ячейке global configuration.

### Пути

```python
DATA_DIR = Path('/content/drive/MyDrive/RAG')
SAVE_DIR = Path('/content/alfa_rag_qwen_cache')
```

### Dev-режим

```python
DEV_N = None
```

Для быстрого теста:

```python
DEV_N = 100
```

### Пересборка кэшей

```python
FORCE_REBUILD_INDEX = False
FORCE_REBUILD_CHUNKS = False
FORCE_REGENERATE = False
```

При изменении preprocessing/chunking/retrieval лучше использовать:

```python
FORCE_REBUILD_CHUNKS = True
FORCE_REBUILD_INDEX = True
FORCE_REGENERATE = True
```

### Модели

```python
EMBEDDING_MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'
RERANKER_MODEL_NAME = 'Qwen/Qwen3-Reranker-0.6B'
GENERATOR_MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
```

---

## 8. Архитектура пайплайна

Общий flow:

```text
questions.csv
    ↓
query normalization / intent detection / query variants
    ↓
websites.csv
    ↓
cleaning / section-aware chunking / source classification
    ↓
dense index + BM25 index
    ↓
hybrid retrieval
    ↓
Qwen3 reranking
    ↓
evidence selection
    ↓
grounding / answerability
    ↓
Qwen2.5 generation
    ↓
post-processing / validation
    ↓
submission.csv
```

---

## 9. Retrieval

Retrieval-блок комбинирует dense search и BM25.

Dense search строится на эмбеддингах:

```text
Qwen/Qwen3-Embedding-0.6B
```

Для индекса используется FAISS. BM25 реализован через `rank-bm25`.

Результаты разных retrieval streams объединяются через Reciprocal Rank Fusion.

Пайплайн использует:

* semantic search;
* lexical BM25 search;
* query variants;
* source-type weights;
* intent-aware boosts;
* retrieval diagnostics.

---

## 10. Reranking

После retrieval кандидаты передаются в reranker:

```text
Qwen/Qwen3-Reranker-0.6B
```

Reranker получает компактное представление чанка:

```text
Заголовок
Раздел
Тип страницы
Текст чанка
```

После reranking сохраняются признаки:

* raw reranker score;
* sigmoid score;
* hybrid score;
* final score;
* top titles;
* source types;
* evidence statistics.

Эти признаки используются для диагностики и настройки последующих экспериментов.

---

## 11. Grounding и answerability

Grounding-блок отвечает за то, чтобы генератор получил не просто набор случайных чанков, а полезный evidence pack.

Он собирает:

* прямые факты;
* инструкции и шаги;
* сроки;
* лимиты;
* комиссии;
* исключения;
* условия применимости;
* fallback-информацию о поддержке.

Answerability-блок принимает решение, можно ли отвечать по найденному контексту.

Используются режимы:

```text
direct_supported
general_supported
weak_but_usable
no_support
```

Если evidence отсутствует, итоговый ответ строго:

```text
Нет ответа.
```

Если вопрос персональный, но в базе есть общая инструкция, система может дать общий ответ без утверждения о конкретном статусе клиента.

---

## 12. Генерация ответов

Генерация выполняется моделью:

```text
Qwen/Qwen2.5-7B-Instruct
```

Генератор получает:

* исходный вопрос;
* evidence pack;
* answerability decision;
* ограничения по длине;
* запрет на неподтверждённые факты.

Пайплайн контролирует:

* чтобы ответ был на русском языке;
* чтобы не добавлялись новые числа, сроки, комиссии и условия;
* чтобы no-answer был строго `Нет ответа.`;
* чтобы ответ не ссылался на “контекст” или “фрагменты”;
* чтобы финальный текст был пригоден для submission.

---

## 13. Diagnostics

Вместе с submission сохраняется diagnostics-файл.

Он помогает анализировать:

* долю no-answer;
* длины ответов;
* повторяющиеся ответы;
* выбранные source types;
* reranker scores;
* evidence length;
* режим ответа;
* потенциальные ошибки answerability;
* качество retrieval на разных типах запросов.

Пример базовых проверок:

```python
diag_df['mode'].value_counts()
diag_df['answer_len'].describe()
diag_df['is_no_answer'].mean()
```

---

## 14. Рекомендованный порядок экспериментов

### Шаг 1. Проверка запуска

```python
DEV_N = 50
```

Цель — убедиться, что все модели загружаются, кэши создаются, submission сохраняется.

### Шаг 2. Малый dev-прогон

```python
DEV_N = 300
```

Проверить:

* no-answer rate;
* среднюю длину ответа;
* повторы;
* примеры evidence;
* diagnostics.

### Шаг 3. Полный прогон

```python
DEV_N = None
FORCE_REBUILD_CHUNKS = True
FORCE_REBUILD_INDEX = True
FORCE_REGENERATE = True
```

### Шаг 4. Анализ результата

После получения submission рекомендуется посмотреть:

```python
submission['answer_new'].str.len().describe()
(submission['answer_new'] == 'Нет ответа.').mean()
submission['answer_new'].duplicated().mean()
```

---

## 15. Производительность

Ориентировочные требования:

```text
Python: 3.10+
GPU: CUDA-compatible
RAM: 16 GB+
VRAM: желательно 16–24 GB+
```

Рекомендуемая среда:

```text
Google Colab L4 / A100
```

На T4 возможен запуск, но генерация 6977 ответов может занять значительно больше времени.

---

## 16. Ускорение запуска

Для ускорения можно:

1. Уменьшить dev subset:

```python
DEV_N = 100
```

2. Уменьшить число новых токенов генератора:

```python
GENERATION_CONFIG['max_new_tokens'] = 180
```

3. Уменьшить answer caps:

```python
GENERATION_CONFIG['answer_cap_default'] = 700
```

4. Использовать уже построенные кэши:

```python
FORCE_REBUILD_CHUNKS = False
FORCE_REBUILD_INDEX = False
FORCE_REGENERATE = False
```

---

## 17. Типичные проблемы

### CUDA out of memory

Решения:

```python
GENERATION_CONFIG['max_new_tokens'] = 180
RERANKER_CONFIG['input_top_n'] = 80
RERANKER_CONFIG['rerank_top_k'] = 32
```

Также можно перезапустить runtime и очистить кэши GPU.

### Очень медленная генерация

Решения:

* уменьшить `max_new_tokens`;
* уменьшить `DEV_N` для тестов;
* запускать полный inference на L4/A100;
* не пересобирать индексы при каждом запуске.

### Submission содержит много `Нет ответа.`

Проверить diagnostics:

* retrieval top titles;
* reranker scores;
* evidence length;
* answerability mode;
* source types.

Высокий no-answer rate чаще всего связан не с генератором, а с retrieval/grounding/answerability.

### Submission не проходит формат

Проверить:

```python
assert set(submission.columns) == {'q_id', 'answer_new'}
assert submission['q_id'].is_unique
assert submission['answer_new'].notna().all()
```

---

## 18. Ограничения

* Пайплайн не использует закрытые API генерации.
* Все inference-вызовы выполняются локально.
* Дополнительные закрытые данные не используются.
* `sample_submission.csv` используется только как пример формата.
* Качество зависит от доступного GPU, настроек генерации и полноты retrieval.

---

## 19. Структура ноутбука

Ноутбук состоит из следующих логических ячеек:

```text
1. Установка зависимостей
2. Google Drive / paths
3. Imports / seed / device
4. Global configuration
5. Data loading
6. Preprocessing utilities
7. Section-aware chunking
8. Tokenization / BM25
9. Qwen embedding model / FAISS index
10. Query normalization / intent / variants
11. Hybrid retrieval with RRF
12. Qwen3 reranker
13. Evidence selection and compression
14. Generator-first Qwen answerer
15. End-to-end answer function
16. Generation loop with checkpointing
17. Build and validate submission
18. Lightweight diagnostics / examples
```

---

## 20. Итог

Этот notebook реализует полный generator-first RAG pipeline для задачи Альфа-Банка:

```text
cleaning → chunking → retrieval → reranking → grounding → generation → submission
```

Главные инженерные акценты:

* не копировать `sample_submission`;
* не сводить задачу к extractive QA;
* улучшать retrieval и grounding;
* генерировать ответы только по найденному evidence;
* строго контролировать no-answer;
* сохранять diagnostics для последующих абляций.

Финальный результат — CSV-файл с колонками:

```text
q_id, answer_new
```

готовый для отправки на платформу.
