import hydra
from omegaconf import dictconfig
from sentence_transformers import SentenceTransformer
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
from bertopic.backend import BaseEmbedder
from bertopic.cluster import BaseCluster
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from glob import glob
from typing import Union
from bertopic import BERTopic

random.seed(25)


class SupervisedTopic:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_file_names(self, data_path: Union[str, Path], extension=".txt"):
        file_names = glob(
            str(self.cfg.project_root / data_path) + f"/**/*{extension}",
            recursive=True,
        )
        return file_names

    def run(self):
        model_path = Path(self.cfg.project_root) / self.cfg.model_path
        sentence_model = SentenceTransformer(model_path)
        file_names = self.get_file_names(self.cfg.data_path)
        random.shuffle(file_names)
        files = [self.read_file(f) for f in file_names]
        train_files, other_files = train_test_split(files, train_size=0.7)
        eval_files, test_files = train_test_split(other_files, train_size=0.5)

        empty_embedding_model = BaseEmbedder()
        empty_dimensionality_model = BaseDimensionalityReduction()
        empty_cluster_model = BaseCluster()
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        topic_model = BERTopic(
            embedding_model=empty_embedding_model,
            umap_model=empty_dimensionality_model,
            hdbscan_model=empty_cluster_model,
            ctfidf_model=ctfidf_model,
        )
        topics, probs = topic_model.fit_transform


@hydra.main(config_path="config", config_name="supervised_topic")
def hydra_start(cfg: dictconfig):
    st = SupervisedTopic(cfg)
    # mlm.run()
    st.run()


if __name__ == "__main__":
    hydra_start()
