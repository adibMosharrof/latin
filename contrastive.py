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
from sentence_transformers import (
    SentenceTransformer,
    losses,
    models,
    CrossEncoder,
    util,
)
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
        self.log = logging.getLogger(__name__)

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

    def test(self, train_examples, test_examples, embedding_model, label_map):

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
        for item in train_examples:
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
        test_docs, test_y = [], []
        for item in train_examples:
            test_docs.append(item.text)
            test_y.append(label_map[item.label])
        test_y_mapped = []
        for val in test_y:
            try:
                test_y_mapped.append(mappings[val])
            except KeyError:
                test_y_mapped.append(-1)
        topic_distr, topic_token_distr = topic_model.approximate_distribution(
            # docs, batch_size=1000, calculate_tokens=True, use_embedding_model=True
            test_docs,
            calculate_tokens=False,
            use_embedding_model=True,
        )
        log = logging.getLogger(__name__)
        for k in [1, 2, 3, 4, 5]:
            acc = top_k_accuracy_score(test_y_mapped, topic_distr, k=k)
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

    def mean_precision_at_k(self, actual_results, retrieved_results, k=1):
        score = []
        for true, pred in zip(actual_results, retrieved_results):
            score.append(self.precision_at_k(true, pred, k))
        return np.mean(score)

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
        a = 1

    def get_mean_precisions(self, reference, preds):
        for k in [1, 2, 3, 4, 5]:
            acc = self.mean_precision_at_k(reference, preds, k=k)
            acc_str = f"Top {k} accuracy: {acc}"
            self.log.info(acc_str)
            print(acc_str)

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
                str(self.cfg.project_root / self.cfg.model_path),
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
        self.test_semantic(
            # embedding_model=model,
            # embedding_model=str(self.cfg.project_root / self.cfg.model_path),
            embedding_model="silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-medieval-latin",
            train_examples=train_examples,
            test_examples=test_examples,
            label_map=label_map,
        )

        # self.test(
        #     embedding_model=model,
        #     train_examples=train_examples,
        #     test_examples=test_examples,
        #     label_map=label_map,
        # )


@hydra.main(config_path="config", config_name="contrastive")
def hydra_start(cfg: DictConfig):
    c = Contrastive(cfg)
    # c.run()
    c.old_run()


if __name__ == "__main__":
    hydra_start()
