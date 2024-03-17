from pathlib import Path
from dotmap import DotMap
from glob import glob

# from torch.utils.data import Dataset
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from lat_bert import LatinBERT
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import random

random.seed(25)

# class MlmDataset(Dataset):

#     def __init__(
#         self,
#         file_paths: list[str],
#     ):
#         self.file_paths = file_paths

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx) -> str:
#         with open(self.file_paths[idx], "r") as f:
#             text = f.read()
#         return text


class Mlm:
    def __init__(self, cfg):
        self.cfg = cfg
        self.block_size = 256

    def get_file_names(self):
        files = glob(str(self.cfg.data_path) + "/**/*.txt", recursive=True)
        return files
        patterns = ["/**/*.txt", "/*.txt"]
        all_files = []
        for path, pattern in zip(self.cfg.data_path, patterns):
            all_files += glob(str(path) + pattern)
        return all_files

    def read_file(self, file: str):
        with open(file, "r") as f:
            text = f.read()
        return text

    def read_file_as_dict(self, file: str):
        with open(file, "r") as f:
            text = f.read()
        return {"text": text}

    def get_tokenized_text(self, files: list[str], tokenizer):
        out = []
        for file in files:
            text = self.read_file(file)
            toks = tokenizer.tokenize(text)
            out.append(toks)
        return out

    def preprocess_function(self, rows):
        return self.tokenizer([t for t in rows["text"]])

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of block_size.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def run(self):
        all_files_names = self.get_file_names()
        file_names = (
            all_files_names[: self.cfg.num_files]
            if self.cfg.num_files
            else all_files_names
        )

        # bert = LatinBERT(tokenizerPath=cfg.tokenizer_path, bertPath=cfg.bert_path)
        model = AutoModelForMaskedLM.from_pretrained(self.cfg.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.bert_path)
        # dataset = MlmDataset(files_names)
        texts = [self.read_file_as_dict(file) for file in file_names]
        ds = Dataset.from_list(texts)
        tokenized_ds = ds.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.cfg.num_proc,
            remove_columns=ds.column_names,
        )
        latin_dataset = tokenized_ds.map(
            self.group_texts,
            batched=True,
            num_proc=self.cfg.num_proc,
        )
        ds_with_splits = latin_dataset.train_test_split(test_size=0.2)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=0.15
        )
        training_args = TrainingArguments(
            output_dir="outputs",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=5,
            weight_decay=0.01,
            per_device_train_batch_size=self.cfg.train_batch_size,
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            logging_steps=10,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_with_splits["train"],
            eval_dataset=ds_with_splits["test"],
            data_collator=data_collator,
        )
        trainer.train()
        model.save_pretrained("results")


if __name__ == "__main__":
    cfg = DotMap(
        # data_path=[Path("data/keyed/canon"), Path("data/unkeyed/canon")],
        data_path=[Path("data/"), Path("data/cltk/lat/text/lat_text_latin_library")],
        # num_files=50,
        num_files=None,
        num_proc=8,
        bert_path=Path("models/latin_bert"),
        tokenizer_path=Path("models/latin_bert/latin.subword.encoder"),
        train_batch_size=45,
        eval_batch_size=100,
        gradient_accumulation_steps=1,
    )

    mlm = Mlm(cfg)
    mlm.run()
