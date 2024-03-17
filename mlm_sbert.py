from glob import glob
import json
from pathlib import Path
import re
from typing import Union
from dotmap import DotMap
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
import random
from sklearn.model_selection import train_test_split
import os
from transformers.trainer_callback import EarlyStoppingCallback
import torch
from transformers.trainer_utils import IntervalStrategy

random.seed(25)


class TokenizedDataset:
    def __init__(self, tokenizer, files: list[str], max_length=512):
        self.tokenizer = tokenizer
        self.files = files
        self.max_length = max_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> str:
        text = self.files[idx]
        return self.tokenizer(
            text,
            # return_tensors="pt",
            max_length=self.max_length,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_special_tokens_mask=True,
        )


class MlmSbert:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.project_root = Path(self.cfg.project_root)

    def read_file(self, file: str):
        file_path = Path(file)
        with open(file_path, "r") as f:
            text = f.read()
            text = text.strip()
            if file_path.suffix == ".tess":
                text = re.sub(r"<[^>]*>", "", text)
        return text

    def get_file_names(self, data_path: Union[str, Path], extension=".txt"):
        file_names = glob(
            str(self.cfg.project_root / data_path) + f"/**/*{extension}",
            recursive=True,
        )
        return file_names

    def train_model(
        self, epochs: int, file_names: list[str], model_path: str, save_path: str
    ):
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        random.shuffle(file_names)
        files = [self.read_file(f) for f in file_names]
        train_files, other_files = train_test_split(files, train_size=0.7)
        eval_files, test_files = train_test_split(other_files, train_size=0.5)
        train_dataset = TokenizedDataset(
            tokenizer, train_files, max_length=self.cfg.max_length
        )
        eval_dataset = TokenizedDataset(
            tokenizer, eval_files, max_length=self.cfg.max_length
        )
        test_dataset = TokenizedDataset(
            tokenizer, test_files, max_length=self.cfg.max_length
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=self.cfg.mlm_probability
        )
        deepspeed_path = str(self.cfg.project_root / "config" / "ds_config.json")
        training_args = self.get_training_args(epochs)
        # training_args = TrainingArguments(
        #     output_dir="outputs",
        #     metric_for_best_model="eval_loss",
        #     evaluation_strategy=IntervalStrategy.STEPS,
        #     save_steps=self.cfg.save_steps,
        #     eval_steps=self.cfg.eval_steps,
        #     save_total_limit=3,
        #     load_best_model_at_end=True,
        #     learning_rate=2e-5,
        #     num_train_epochs=epochs,
        #     weight_decay=0.01,
        #     warmup_steps=200,
        #     dataloader_num_workers=8,
        #     report_to="wandb",
        #     dataloader_drop_last=True,
        #     fp16=self.cfg.fp16,
        #     fp16_full_eval=self.cfg.fp16,
        #     per_device_train_batch_size=self.cfg.train_batch_size,
        #     per_device_eval_batch_size=self.cfg.eval_batch_size,
        #     gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
        #     logging_steps=50,
        #     deepspeed=deepspeed_path,
        # )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.cfg.early_stopping_patience
                ),
            ],
        )
        trainer.train()
        model.save_pretrained(save_path)
        # tokenizer.save_pretrained(save_path)

    def pretrain(self):
        pretrain_path = Path(self.cfg.project_root) / self.cfg.model_name
        if pretrain_path.exists():
            return pretrain_path
        all_file_names = []
        for dp, file_extension, split_percent in self.cfg.pretrain_data:
            file_names = self.get_file_names(dp, file_extension)
            file_names = file_names[: int(len(file_names) * split_percent)]
            all_file_names += file_names

        self.train_model(
            self.cfg.pretrain_epochs,
            all_file_names,
            self.cfg.model_name,
            self.cfg.pretrain_out_path,
        )
        return self.cfg.pretrain_out_path

    def train(self):
        file_names = self.get_file_names(self.cfg.data_path)
        file_names = file_names[: int(len(file_names) * self.cfg.data_split_percentage)]
        self.train_model(
            self.cfg.train_epochs,
            file_names,
            self.cfg.pretrain_out_path,
            self.cfg.train_out_path,
        )

    def get_training_args(self, epochs):
        bf16 = False
        fp16 = False
        if self.cfg.fp16:
            if torch.cuda.is_bf16_supported():
                bf16 = True
            else:
                fp16 = True
        return TrainingArguments(
            output_dir="outputs",
            metric_for_best_model="eval_loss",
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=self.cfg.save_steps,
            eval_steps=self.cfg.eval_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            learning_rate=2e-5,
            num_train_epochs=epochs,
            weight_decay=0.01,
            warmup_steps=300,
            dataloader_num_workers=8,
            report_to="wandb",
            dataloader_drop_last=True,
            fp16=fp16,
            fp16_full_eval=fp16,
            bf16=bf16,
            bf16_full_eval=bf16,
            per_device_train_batch_size=self.cfg.train_batch_size,
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            logging_steps=50,
        )

    def run(self):
        self.pretrain()
        self.train()
        print(os.getcwd())


@hydra.main(config_path="config", config_name="mlm_bert")
def hydra_start(cfg: DictConfig):
    mlm = MlmSbert(cfg)
    mlm.run()


if __name__ == "__main__":
    hydra_start()
