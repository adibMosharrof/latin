import logging
from pathlib import Path
import random

from dotmap import DotMap
import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from sentence_transformers import SentenceTransformer, util, models
from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
from sklearn.model_selection import train_test_split


random.seed(25)


class Inference:
    def __init__(self, cfg):
        cfg.project_root = Path(cfg.project_root)
        self.cfg = cfg
        self.log = logging.getLogger(__name__)

    def get_texts(self):
        canon_csv_path = (
            Path(self.cfg.project_root) / "data" / "data_stats" / "filtered_canons.csv"
        )
        canon_csv = pd.read_csv(canon_csv_path)
        canons = canon_csv["canon"].tolist()
        all_text = []
        for canon in canons:
            canon_path = Path(self.cfg.project_root) / self.cfg.data_path / canon
            files = list(canon_path.glob("*.txt"))
            for file in files:
                with open(file, "r") as f:
                    text = f.read()
                    all_text.append(DotMap(text=text.strip(), label=canon))
        return all_text

    def get_label_map(self, texts):
        label_map = {}
        labels_set = set([t.label for t in texts])
        for i, label in enumerate(labels_set):
            label_map[label] = i
        return label_map

    def run(self):
        texts = self.get_texts()
        texts = texts[: int(len(texts) * self.cfg.data_split_percentage)]
        label_map = self.get_label_map(texts)
        train_examples, other_examples = train_test_split(texts, train_size=0.7)
        eval_examples, test_examples = train_test_split(other_examples, train_size=0.5)

        model_path = self.cfg.project_root / self.cfg.model_path
        word_embedding_model = models.Transformer(
            model_path,
            max_seq_length=self.cfg.max_length,
            tokenizer_name_or_path=model_path,
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )
        # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        bi_encoder.max_seq_length = 512
        cross_encoder = CrossEncoder(model_path)
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
        corpus_labels = [label_map[item.label] for item in train_examples]
        for query, hits in zip(test_examples, all_hits):
            cross_inp = [
                [query.text, train_examples[hit["corpus_id"]].text] for hit in hits
            ]
            cross_scores = cross_encoder.predict(cross_inp)
            for idx in range(len(cross_scores)):
                hits[idx]["cross-score"] = cross_scores[idx]
                hits[idx]["corpus_label"] = corpus_labels[hits[idx]["corpus_id"]]

        corpus_labels = []
        for hits in all_hits:
            sorted(hits, key=lambda x: x["score"])
            corpus_labels.append([hit["corpus_label"] for hit in hits])
        print("Scores with retrieval")
        self.log.info("Scores with retrieval")
        self.get_mean_precisions(test_labels, corpus_labels)
        for hits in all_hits:
            sorted(hits, key=lambda x: x["cross-score"])

        print("Scores with re-rank")
        self.log.info("Scores with re-rank")
        self.get_mean_precisions(test_labels, corpus_labels)

    def get_mean_precisions(self, reference, preds):
        for k in [1, 2, 3, 4, 5]:
            acc = self.mean_precision_at_k(reference, preds, k=k)
            acc_str = f"Top {k} accuracy: {acc}"
            self.log.info(acc_str)
            print(acc_str)

    def mean_precision_at_k(self, actual_results, retrieved_results, k=1):
        score = []
        for true, pred in zip(actual_results, retrieved_results):
            score.append(self.precision_at_k(true, pred, k))
        return np.mean(score)

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


@hydra.main(config_path="config", config_name="inference")
def hydra_start(cfg: DictConfig):
    inf = Inference(cfg)
    inf.run()


if __name__ == "__main__":
    hydra_start()
