# Безумный Лис Бот  ![frantics_fox](https://drive.google.com/uc?export=view&id=1WL-LT3xLcVX-z0RhIGpiktsE6QtomuyG)

Это проект по созданию БЕЗУМНОГО ЛИСА. ЛИС — это не просто бездушная машина. Это гениальная персона, способная найти общий с вами язык, каким бы грязным он ни был. Просто сохраните ваш телеграм чат в формате `json`, обучите на нём ЛИСА и смотрите, как он начинает повторять за вами...

## Версии
В папке `./models/` можно найти нескольких моделей, реализающих данного бота:
- `seq2seq` — seq2seq модель с применением Luong Attention Mechanism. Бот создан на основе [данного туториала](https://docs.pytorch.org/tutorials/beginner/chatbot_tutorial)
- `transformer` — файн-тьюнинг [ai-forever/rugpt3small_based_on_gpt2](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2)

## Обучение
Для обучения моделей сохраните ваш телеграм чат без сохранения медиа:

`Три точки в верхнем правом углу` -> `Экспорт истории чата` -> `Формат: Машиночитаемый JSON` -> `Экспортировать`

На выходе вы получите `json`-файл для обучения

- Для обучения `seq2seq` модели используйте метод `train()` класса `Seq2SeqChatbot`. Укажите путь к `json`-файлу в переменной `CHAT_HISTORY_PATH`. Запустите файл `./models/seq2seq/model.py` как модуль. Для этого в корне проекта выполните команду `python -m models.seq2seq.model`.

- Для обучения `transformer` укажите путь к `json`-файлу в методе `prepare_data()` класса `FineTuner` в файле `./models/transformer/finetuner.py`. 
### Пример:

```python
finetuner = FineTuner()
dataset_path = finetuner.prepare_data()
finetuner.fine_tune(dataset_path, output_name='fine_tuned_model_gpt_2')
```

Удобнее всего обучение производить в `ipynb`-блокнотах:
- [seq2seq](https://www.youtube.com/watch?v=dQw4w9WgXcQ)
- [transformer](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

## Запуск
Зависимости для сервера и телеграм-бота распологаются в `./requirements.txt`

Зависимости для каждой модели распологаются в `./models/<НАЗВАНИЕ_МОДЕЛИ>/requirements.txt`

Существует 2 способа запуска бота:
### СПОСОБ 1 (Локальный запуск)
  - Настроить модель в файле `./bot_local.py` (на данный момент стоит seq2seq-версия)
  - Указать значения следующих переменных среды:
    - `TOKEN` — токен телеграм-бота
    - `BOT_USERNAME` — id бота
    - `CHAT_ID` — id чата, в котором находится бот
  - В случае использования seq2seq модели, указать в `CHECKPOINT_PATH` путь к архиву с чекпоинтом
  - Выполнить команду: `python bot_local.py`
### СПОСОБ 2 (Сервер)
  - Склонировать репозиторий на удаленный сервер
  - В `./server.py` указать нужную модель (на данный момент стоит transformer-версия)
  - Запустить сервер командой `uvicorn server:app --host 0.0.0.0 --port "<ПОРТ>"`
  - Также можно использовать данный Dockerfile (настроен для работы на huggingface spaces):

```Dockerfile
FROM python:3.12

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
RUN pip install --no-cache-dir -r models/transformer/requirements.txt
RUN pip install --no-cache-dir -r models/seq2seq/requirements.txt

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

  - В файле `./bot_server.py` указать значения переменных среды (аналогично способу 1), а в функции `handel_response()` указать ссылку на веб-сервер

## Результат
Теперь, когда ЛИС полностью прознал все ваши секреты, которыми вы так бездумно делились в ваших чатах, пришло время сидеть и наслаждаться пёрлами ЛИСА
