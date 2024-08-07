from dataclasses import dataclass
import hydra
from omegaconf import DictConfig
import os
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, losses, util
from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
from sentence_transformers.readers import InputExample
from canon_trainer import CanonTrainer
from create_unkeyed_data import CreateUnkeyedData
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from bidict import bidict
from typing import Optional


@dataclass
class UnkeyedData:
    file_name: str
    text: str
    label: Optional[int] = None
    canon: Optional[str] = None


class UnkeyedTrainer(CanonTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg.data_path = self.cfg.project_root / self.cfg.data_path

    def get_unkeyed_test_data(self, test_data):
        unkeyed_data = []
        root_path = self.cfg.data_path / "unkeyed" / self.cfg.data_type
        for i, row in test_data.iterrows():
            with open(root_path / row.file_name, "r") as f:
                text = f.read()
            unkeyed_data.append(UnkeyedData(file_name=row.file_name, text=text))
        return unkeyed_data

    def train_files_with_content(self, train_data, label_map):
        out = []
        for i, row in train_data.iterrows():
            with open(
                self.cfg.data_path
                / "keyed"
                / self.cfg.data_type
                / row.canon
                / row.file_name,
                "r",
            ) as f:
                text = f.read()
            out.append(
                UnkeyedData(
                    text=text,
                    file_name=row.file_name,
                    label=label_map[row.canon],
                    canon=row.canon,
                )
            )
        return out

    def get_train_input_examples(self, train_data, label_map):
        out = []
        for i, row in train_data.iterrows():
            with open(
                self.cfg.data_path
                / "keyed"
                / self.cfg.data_type
                / row.canon
                / row.file_name,
                "r",
            ) as f:
                text = f.read()
            label = row.canon
            out.append(InputExample(texts=[text], label=label_map[label]))
        return out

    def get_label_map(self, train_data):
        label_map = {}
        labels_set = train_data.canon.unique()
        for i, label in enumerate(labels_set):
            label_map[label] = i
        return bidict(label_map)

    def contrastive_model(self, train_data, label_map):
        train_input_examples = self.get_train_input_examples(train_data, label_map)
        train_input_examples = train_input_examples[
            : int(len(train_input_examples) * self.cfg.train_data_split_percentage)
        ]
        model = self.get_contrastive_model()
        train_loss = losses.BatchAllTripletLoss(model=model)
        train_dl = DataLoader(
            train_input_examples, batch_size=self.cfg.train_batch_size, shuffle=True
        )
        out_path = str(Path(os.getcwd()) / self.cfg.out_path)
        model.fit(
            train_objectives=[(train_dl, train_loss)],
            epochs=self.cfg.epochs,
            # evaluator=evaluator,
            evaluation_steps=self.cfg.eval_steps,
            warmup_steps=100,
            output_path=out_path,
        )
        model.save(out_path)
        return out_path

    def test_semantic(self, train_examples, test_examples, embedding_model, label_map):
        bi_encoder = SentenceTransformer(embedding_model)
        bi_encoder.max_seq_length = 512
        cross_encoder = CrossEncoder(embedding_model)
        top_k = 32
        corpus_embeddings = bi_encoder.encode(
            [item.text for item in train_examples],
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        test_embeddings = bi_encoder.encode(
            [item.text for item in test_examples],
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        all_hits = util.semantic_search(test_embeddings, corpus_embeddings, top_k=top_k)
        out = []
        for hits, test_example in zip(all_hits, test_examples):
            sorted(hits, key=lambda x: x["score"])
            row = {
                "test_file_name": test_example.file_name,
                "test_text": test_example.text,
            }
            for i, hit in enumerate(hits[:5]):

                prefix = f"top-{i+1}_"
                train_example = train_examples[hit["corpus_id"]]
                row.update(
                    {
                        prefix + f"{self.cfg.data_type}_pred": train_example.canon,
                        prefix + "prob": round(hit["score"], 4),
                        prefix + "train_file": train_example.file_name,
                        prefix + "train_text": train_example.text,
                    }
                )
            out.append(row)

        out_path = "unkeyed_results.csv"
        out_df = pd.DataFrame(out)
        out_df.to_csv(out_path)

    def contrastive_stage(self, train_data, label_map):
        if self.cfg.contrastive_model_path:
            return str(self.cfg.project_root / self.cfg.contrastive_model_path)
        return self.contrastive_model(train_data, label_map)

    def run(self):
        if not (self.cfg.project_root / self.cfg.train_out_path).exists():
            cd = CreateUnkeyedData(self.cfg)
            cd.run()
        train_data = pd.read_csv(self.cfg.project_root / self.cfg.train_out_path)
        test_data = pd.read_csv(self.cfg.project_root / self.cfg.test_out_path)
        test_data = test_data[
            : int(len(test_data) * self.cfg.train_data_split_percentage)
        ]
        label_map = self.get_label_map(train_data)
        train_files_with_content = self.train_files_with_content(train_data, label_map)
        unkeyed_test_data = self.get_unkeyed_test_data(test_data)
        model_path = self.contrastive_stage(train_data, label_map)
        self.test_semantic(
            train_files_with_content, unkeyed_test_data, model_path, label_map
        )


@hydra.main(config_path="config", config_name="unkeyed_trainer")
def hydra_start(cfg: DictConfig):
    trainer = UnkeyedTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    hydra_start()
