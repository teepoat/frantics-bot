from pathlib import Path
from .utils import modified_tokenizer
from .telegram_data_extractor import TelegramDataExtractor
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from .constants import CHECKPOINT_PATH


class FineTuner:
    def __init__(self,
                 model_name="ai-forever/rugpt3small_based_on_gpt2",
                 cache_dir="model_cache",
                 data_path=CHECKPOINT_PATH):
        self.data_path = Path(data_path)

        # Инициализация токенизатора и модели
        self.tokenizer = modified_tokenizer(model_name, cache_dir, self.data_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=str(self.data_path / cache_dir))

    def prepare_data(self):
        """
        Подготовка данных для обучения
        """
        messages = TelegramDataExtractor.load_messages_from_json("/kaggle/input/chat-history/chat_history_small.json")
        dataset_path = TelegramDataExtractor.conversations_from_messages(self.data_path, self.tokenizer, messages)
        return dataset_path

    def fine_tune(self,
                  dataset_path,
                  output_name='fine_tuned_model',
                  num_train_epochs=10,
                  per_device_train_batch_size=8,
                  learning_rate=5e-5,
                  save_steps=10_000):
        """
        Дообучение модели на заданном датасете.
        """
        dataset = load_dataset("text", data_files={"train": "train_dataset.txt"})

        def preprocess(example):
            # Tokenize while preserving structure
            return self.tokenizer(example["text"], truncation=True, max_length=300)
        
        train_dataset = dataset.map(preprocess, batched=True)["train"]

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir=str(self.data_path / output_name),
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            # fp16=True,
            # gradient_accumulation_steps=2,
            save_steps=save_steps,
            learning_rate=learning_rate,
            torch_compile=True,
            save_total_limit=2,
            logging_dir=str(self.data_path / 'logs'),
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        trainer.train()
        # Сохранение обученной модели и токенизатора
        self.model.save_pretrained(str(self.data_path / output_name))
        self.tokenizer.save_pretrained(str(self.data_path / output_name))
