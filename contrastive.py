import itertools
from dotmap import DotMap
import hydra
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from glob import glob
from sentence_transformers.readers import InputExample
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    LabelAccuracyEvaluator,
)
from sentence_transformers import SentenceTransformer, losses, models
from bertopic.backend import BaseEmbedder
from bertopic.cluster import BaseCluster
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import json
from sklearn.metrics import top_k_accuracy_score
import logging


class Contrastive:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.project_root = Path(self.cfg.project_root)

    def get_stop_words(self):
        path = self.cfg.project_root / "data" / "stopwords_latin.json"
        with open(path, "r") as f:
            stop_words = json.load(f)
        all_words = []
        for k, v in stop_words.items():
            if type(v) == list:
                all_words += v
            else:
                for k1, v1 in v.items():
                    if type(v1) == list:
                        all_words += v1
                    else:
                        raise ValueError("stop words must be a list")

        special_files = ["do_not_index.txt", "stopwords.txt"]
        for file_name in special_files:
            with open(
                Path(self.cfg.project_root / "special_words" / file_name), "r"
            ) as f:
                words = f.readlines()
                words = [w.strip() for w in words]
                all_words += words

        return list(set(all_words))

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

    def get_label_map(self, texts):
        label_map = {}
        labels_set = set([t.label for t in texts])
        for i, label in enumerate(labels_set):
            label_map[label] = i
        return label_map

    def test(self, test_examples, embedding_model, label_map):

        # empty_embedding_model = BaseEmbedder()
        empty_dimensionality_model = BaseDimensionalityReduction()
        empty_cluster_model = BaseCluster()
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        stop_words = self.get_stop_words()
        vectorizer_model = CountVectorizer(
            stop_words=stop_words, min_df=2, ngram_range=(1, 2)
        )
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=empty_dimensionality_model,
            hdbscan_model=empty_cluster_model,
            ctfidf_model=ctfidf_model,
            vectorizer_model=vectorizer_model,
        )

        docs, y = [], []
        value_map = {value: key for key, value in label_map.items()}
        for item in test_examples:
            docs.append(item.text)
            y.append(label_map[item.label])
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings, y=y)
        mappings = topic_model.topic_mapper_.get_mappings()
        y_mapped = [mappings[val] for val in y]
        mappings = {value: value_map[key] for key, value in mappings.items()}

        topic_model.set_topic_labels(mappings)
        # Assign original classes to our topics
        df = topic_model.get_topic_info()
        df["Class"] = df.Topic.map(mappings)
        topic_distr, topic_token_distr = topic_model.approximate_distribution(
            # docs, batch_size=1000, calculate_tokens=True, use_embedding_model=True
            docs,
            calculate_tokens=True,
            use_embedding_model=True,
        )
        log = logging.getLogger(__name__)
        for k in [1, 2, 3, 4, 5]:
            acc = top_k_accuracy_score(y_mapped, topic_distr, k=k)
            acc_str = f"Top {k} accuracy: {acc}"
            log.info(acc_str)
            print(acc_str)

        samples = -5

        top_preds = np.argsort(topic_distr, axis=1)[:, samples:]
        top_preds_probs = np.take_along_axis(topic_distr, top_preds, axis=1)
        top_pred_labels = np.vectorize(mappings.get)(top_preds)
        log.info("Example outputs:\n")
        for i in range(5):
            log.info(f"Document name: {test_examples[i].label}")
            log.info(f"True: {mappings[y_mapped[i]]}")
            log.info(f"Predicted: {top_pred_labels[i]}")
            log.info(f"Probabilities: {top_preds_probs[i]}")
            log.info("-" * 25)

    def old_run(self):
        texts = self.get_texts()
        texts = texts[: int(len(texts) * self.cfg.data_split_percentage)]
        label_map = self.get_label_map(texts)
        train_examples, other_examples = train_test_split(texts, train_size=0.7)
        eval_examples, test_examples = train_test_split(other_examples, train_size=0.5)
        train_input_examples = self.get_train_input_examples(train_examples, label_map)
        train_dl = DataLoader(
            train_input_examples, batch_size=self.cfg.train_batch_size, shuffle=True
        )
        if self.cfg.model_path:
            model = SentenceTransformer(
                str(self.cfg.project_root / self.cfg.model_path)
            )
        else:
            word_embedding_model = models.Transformer(
                str(self.cfg.project_root / self.cfg.model_name),
                max_seq_length=self.cfg.max_length,
                tokenizer_name_or_path=self.cfg.tokenizer_name,
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension()
            )
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            train_loss = losses.BatchAllTripletLoss(model=model)
            eval_input_examples = self.get_eval_input_examples(eval_examples)
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                eval_input_examples
            )

            model.fit(
                train_objectives=[(train_dl, train_loss)],
                epochs=self.cfg.epochs,
                evaluator=evaluator,
                evaluation_steps=self.cfg.eval_steps,
                warmup_steps=100,
                output_path=self.cfg.out_path,
            )
        self.test(
            embedding_model=model, test_examples=test_examples, label_map=label_map
        )


@hydra.main(config_path="config", config_name="contrastive")
def hydra_start(cfg: DictConfig):
    c = Contrastive(cfg)
    # c.run()
    c.old_run()


if __name__ == "__main__":
    hydra_start()
