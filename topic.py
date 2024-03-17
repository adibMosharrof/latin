import json
import hydra
from omegaconf import dictconfig
from sentence_transformers import SentenceTransformer
from pathlib import Path
import random
from glob import glob
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

random.seed(25)
import umap
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    OpenAI,
    PartOfSpeech,
)
import numpy as np


class Topic:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.project_root = Path(self.cfg.project_root)

    def read_file(self, file: str):
        with open(file, "r") as f:
            text = f.read()
        return text

    def get_file_names(self):
        files = glob(
            str(self.cfg.project_root / self.cfg.data_path) + "/**/*.txt",
            recursive=True,
        )
        return files

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

    def bert_topic(self):
        stop_words = self.get_stop_words()
        file_names = self.get_file_names()
        file_names = file_names[: int(len(file_names) * self.cfg.data_split_percentage)]
        titles = []
        for name in file_names:
            p = Path(name)
            titles.append(p.parent.name + "/" + p.stem)
        files = [self.read_file(f) for f in file_names]
        embedding_model = SentenceTransformer(self.cfg.model_name)
        embeddings = embedding_model.encode(files, show_progress_bar=True)
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=25,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        vectorizer_model = CountVectorizer(
            stop_words=stop_words, min_df=2, ngram_range=(1, 2)
        )
        keybert_model = KeyBERTInspired()
        mmr_model = MaximalMarginalRelevance(diversity=0.3)
        representation_model = {
            "KeyBERT": keybert_model,
            "MMR": mmr_model,
        }
        topic_model = BERTopic(
            # Pipeline models
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            # representation_model=representation_model,
            # Hyperparameters
            top_n_words=10,
            verbose=True,
        )
        topics, probs = topic_model.fit_transform(files, embeddings)
        # t_info = topic_model.get_topic_info()
        # topic_distr, _ = topic_model.approximate_distribution(files, window=8, stride=4)
        fig = topic_model.visualize_topics(custom_labels=True)
        fig.write_html("topic.html")
        fig_bar = topic_model.visualize_barchart()
        fig_bar.write_html("topic_bar.html")
        # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
        reduced_embeddings = UMAP(
            n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine"
        ).fit_transform(embeddings)
        # Visualize the documents in 2-dimensional space and show the titles on hover instead of the abstracts
        # NOTE: You can hide the hover with `hide_document_hover=True` which is especially helpful if you have a large dataset
        # NOTE: You can also hide the annotations with `hide_annotations=True` which is helpful to see the larger structure
        fig_doc = topic_model.visualize_documents(
            # files,
            titles,
            # titles, reduced_embeddings=reduced_embeddings, custom_labels=True
            # hide_annotations=True,
            # hide_document_hover=True,
            reduced_embeddings=reduced_embeddings,
            # custom_labels=True,
        )
        fig_doc.write_html("topic_doc.html")

        hierarchical_topics = topic_model.hierarchical_topics(files)

        fig_h_topics = topic_model.visualize_hierarchical_documents(
            titles,
            hierarchical_topics,
            reduced_embeddings=reduced_embeddings,
            hide_document_hover=False,
        )
        fig_h_topics.write_html("topic_hierarchical.html")

        df = pd.DataFrame({"Document": files, "FileNames": titles})
        doc_info = topic_model.get_document_info(files, df=df)
        all_topics = np.unique(doc_info.Topic.values)
        out = []
        topic_by_groups = []
        for topic in all_topics[1:]:
            out.append(str(topic))
            topic_group = doc_info[doc_info.Topic == topic]
            topic_group_list = []
            for i, row in topic_group.iterrows():
                out.append(row.FileNames)
                topic_group_list.append(row.FileNames)
            out.append("-" * 50)
            topic_by_groups.append(topic_group_list)
        for i,topic_group in enumerate(topic_by_groups):
            with open(f"topic_{i}.txt","w") as f:
                f.write("\n".join(topic_group))
        with open("topics.txt", "w") as f:
            f.write("\n".join(out))

    def run(self):
        # model_path = self.cfg.project_root / self.cfg.model_path
        model_path = self.cfg.model_path
        model = SentenceTransformer(model_path)
        file_names = self.get_file_names()
        file_names = file_names[: int(len(file_names) * self.cfg.data_split_percentage)]
        files = [self.read_file(f) for f in file_names]
        embeddings = model.encode(files)
        umap_embeddings = umap.UMAP(
            n_neighbors=15, n_components=5, metric="cosine", random_state=25
        ).fit_transform(embeddings)
        cluster = hdbscan.HDBSCAN(
            min_cluster_size=15, metric="euclidean", cluster_selection_method="eom"
        ).fit(umap_embeddings)

        umap_data = umap.UMAP(
            n_neighbors=15, n_components=2, min_dist=0.0, metric="cosine"
        ).fit_transform(embeddings)

        result = pd.DataFrame(umap_data, columns=["x", "y"])
        result["labels"] = cluster.labels_
        fig, ax = plt.subplots(figsize=(20, 10))

        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color="#BDBDBD", s=0.05)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap="hsv_r")
        plt.colorbar()
        plt.savefig("topic.png")


@hydra.main(config_path="config", config_name="topic")
def hydra_start(cfg: dictconfig):
    mlm = Topic(cfg)
    # mlm.run()
    mlm.bert_topic()


if __name__ == "__main__":
    hydra_start()
