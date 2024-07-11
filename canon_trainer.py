import itertools
import os
from pathlib import Path
from dotmap import DotMap
import hydra
import numpy as np
from omegaconf.dictconfig import DictConfig
import pandas as pd
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader

from create_data import CreateData
from sentence_transformers import (
    SentenceTransformer,
    losses,
    models,
    CrossEncoder,
    util,
)
import logging
import evaluate
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from bidict import bidict


class CanonTrainer:

    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.project_root = Path(self.cfg.project_root)
        formatter = logging.Formatter(fmt="%(message)s")
        root_logger = logging.getLogger()  # no name
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(formatter)
        self.log = root_logger

    def test_semantic(
        self, train_examples, test_examples, embedding_model, label_map, num_files
    ):
        self.log.info("Testing with %d files", num_files)
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
        test_labels = [label_map[item.label] for item in test_examples]
        train_corpus_labels = [label_map[item.label] for item in train_examples]
        for query, hits in zip(test_examples, all_hits):
            cross_inp = [
                [query.text, train_examples[hit["corpus_id"]].text] for hit in hits
            ]
            cross_scores = cross_encoder.predict(cross_inp)
            for idx in range(len(cross_scores)):
                hits[idx]["cross-score"] = cross_scores[idx]
                hits[idx]["corpus_label"] = train_corpus_labels[hits[idx]["corpus_id"]]

        corpus_labels = []
        for hits in all_hits:
            sorted(hits, key=lambda x: x["score"])
            corpus_labels.append([hit["corpus_label"] for hit in hits])
        print("Scores with retrieval")
        self.log.info("Scores with retrieval")
        mean_precisions = self.get_mean_precisions(test_labels, corpus_labels)
        for hits in all_hits:
            sorted(hits, key=lambda x: x["cross-score"])

        # print("Scores with re-rank")
        # self.log.info("Scores with re-rank")
        # self.get_mean_precisions(test_labels, corpus_labels)
        self.plot_confusion_matrix(corpus_labels, test_labels, num_files)
        self.per_class_accuracy(
            corpus_labels, test_labels, label_map, mean_precisions, num_files
        )

    def per_class_accuracy(
        self, corpus_labels, test_labels, label_map, mean_precisions, num_files
    ):
        class_wise = defaultdict(list)
        per_class_precisions = []
        for ref, pred in zip(test_labels, corpus_labels):
            class_wise[ref].append(pred)
        for key, value in class_wise.items():
            class_name = label_map.inv[key]
            per_class_precisions.append(
                [class_name] + self.get_mean_precisions([key] * len(value), value)
            )

        column_names = ["class", "1", "2", "3", "4", "5"]
        all_precision = ["all"] + mean_precisions
        df = pd.DataFrame(
            np.concatenate([[all_precision], per_class_precisions]),
            columns=column_names,
        )
        out_dir = Path(os.getcwd()) / self.cfg.out_path
        out_dir.mkdir(exist_ok=True)
        file_name = out_dir / f"per_class_accuracy_num_files_{num_files}.csv"
        df.to_csv(file_name, index=False)
        a = 1

    def plot_confusion_matrix(self, corpus_labels, test_labels, num_files):
        conf_metric = evaluate.load("confusion_matrix")
        top_1_prediction = [l[0] for l in corpus_labels]
        results = conf_metric.compute(
            references=test_labels, predictions=top_1_prediction
        )["confusion_matrix"]
        out_dir = Path(os.getcwd()) / self.cfg.out_path
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / f"confusion_matrix_num_files_{num_files}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(results)
        svm = sns.heatmap(
            # results / np.sum(results), fmt=".2%", cmap="rocket_r", annot=True
            results,
            cmap="rocket_r",
        )
        figure = svm.get_figure()
        figure.savefig(out_dir / f"confusion_matrix_num_files_{num_files}.png")
        # figure.clear()
        # plt.close(figure)
        plt.clf()

    def get_mean_precisions(self, reference, preds):
        out = []
        for k in [1, 2, 3, 4, 5]:
            acc = self.mean_precision_at_k(reference, preds, k=k)
            # acc_str = f"Top {k} accuracy: {acc}"
            # self.log.info(acc_str)
            # print(acc_str)
            out.append(acc)
        return out

    def mean_precision_at_k(self, actual_results, retrieved_results, k=1):
        score = []
        for true, pred in zip(actual_results, retrieved_results):
            score.append(self.precision_at_k(true, pred, k))
        return np.mean(score).round(4)

    def precision_at_k(self, actual_results, retrieved_results, k=1):
        """
        Calculate precision at k.

        Args:
            actual_results (list): List of actual relevant items.
            retrieved_results (list): List of retrieved items.
            k (int): Number of top items to consider.

        Returns:
            float: Precision at k.
        """
        if actual_results is None or retrieved_results is None:
            return 0
        if len(retrieved_results) < k:
            k = len(retrieved_results)
        if k <= 0:
            raise ValueError("k must be a positive integer")
        values_to_consider = retrieved_results[:k]
        result = 0
        if actual_results in values_to_consider:
            result = 1
        return result

    def get_texts(self, canon_data, is_test=False):
        all_text = []
        for i, row in canon_data.iterrows():
            canon_path = (
                Path(self.cfg.project_root)
                / self.cfg.data_path
                / row.canon
                / row.file_name
            )
            with open(canon_path, "r") as f:
                text = f.read()
                label = row.classification_label if is_test else row.canon
                all_text.append(
                    DotMap(text=text.strip(), label=label, num_files=row.num_files)
                )
        return all_text

    def get_train_input_examples(self, rows, label_map):
        return [
            InputExample(texts=[row.text], label=label_map[row.label]) for row in rows
        ]

    def get_eval_input_examples(self, rows):
        comb = list(itertools.combinations(rows, 2))
        out = []
        for c in comb:
            label = 1 if c[0].label == c[1].label else 0
            out.append(InputExample(texts=[c[0].text, c[1].text], label=label))
        return out

    def get_label_map(self, train_texts, test_texts):
        texts = train_texts + test_texts
        label_map = {}
        labels_set = set([t.label for t in texts])
        for i, label in enumerate(labels_set):
            label_map[label] = i
        return bidict(label_map)

    def get_contrastive_model(self):
        word_embedding_model = models.Transformer(
            str(self.cfg.project_root / self.cfg.mlm_model_path),
            max_seq_length=self.cfg.max_length,
            tokenizer_name_or_path=self.cfg.tokenizer_name,
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        return model

    def contrastive_model(self, eval_input_examples, train_input_examples):
        model = self.get_contrastive_model()
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            eval_input_examples
        )
        train_loss = losses.BatchAllTripletLoss(model=model)
        train_dl = DataLoader(
            train_input_examples, batch_size=self.cfg.train_batch_size, shuffle=True
        )
        out_path = str(Path(os.getcwd()) / self.cfg.out_path)
        model.fit(
            train_objectives=[(train_dl, train_loss)],
            epochs=self.cfg.epochs,
            evaluator=evaluator,
            evaluation_steps=self.cfg.eval_steps,
            warmup_steps=100,
            output_path=out_path,
        )
        model.save(out_path)
        return out_path

    def contrastive_stage(self, train_data, test_data) -> str:
        train_texts = self.get_texts(train_data, is_test=True)
        test_texts = self.get_texts(test_data, is_test=True)
        train_texts = train_texts[
            : int(len(train_texts) * self.cfg.train_data_split_percentage)
        ]
        label_map = self.get_label_map(train_texts, test_texts)
        train_examples, eval_examples = train_test_split(train_texts, train_size=0.8)
        train_input_examples = self.get_train_input_examples(train_examples, label_map)
        eval_input_examples = self.get_eval_input_examples(eval_examples)
        if not self.cfg.contrastive_model_path:
            out_path = self.contrastive_model(eval_input_examples, train_input_examples)
        else:
            out_path = str(self.cfg.project_root / self.cfg.contrastive_model_path)
        for i in range(4):
            if i == 1 or i == 2:
                filtered_test = [item for item in test_texts if item.num_files == i]
            else:
                filtered_test = [item for item in test_texts if item.num_files >= i]
            self.test_semantic(
                embedding_model=out_path,
                train_examples=train_examples,
                test_examples=filtered_test,
                label_map=label_map,
                num_files=i,
            )

    def run(self):
        if (self.cfg.project_root / self.cfg.train_out_path).exists():
            cd = CreateData(self.cfg)
            cd.run()
        train_data = pd.read_csv(self.cfg.project_root / self.cfg.train_out_path)
        test_data = pd.read_csv(self.cfg.project_root / self.cfg.test_out_path)
        out_path = self.contrastive_stage(train_data, test_data)


@hydra.main(config_path="config", config_name="canon_trainer")
def hydra_start(cfg: DictConfig):
    trainer = CanonTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    hydra_start()
