from dataclasses import dataclass
from pathlib import Path
from dotmap import DotMap
import numpy as np
import pandas as pd
import os
import random

random.seed(420)
from sklearn.model_selection import train_test_split
from dataclass_csv import DataclassWriter
from create_data import CanonData


class CreateUnkeyedData:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.data_path = self.cfg.project_root / self.cfg.data_path
        self.keyed_path = self.cfg.data_path / "keyed" / self.cfg.data_type
        self.unkeyed_path = self.cfg.data_path / "unkeyed" / self.cfg.data_type

    def create_train_data(self):
        keyed_canons = list(self.keyed_path.iterdir())
        keyed_data = []
        for canon in keyed_canons:
            if not canon.is_dir():
                continue
            files = list(canon.iterdir())
            for file in files:
                keyed_data.append(
                    CanonData(
                        canon=canon.name,
                        file_name=file.name,
                        classification_label=canon.name,
                        num_files=len(files),
                    )
                )

        with open(self.cfg.project_root / self.cfg.train_out_path, "w") as f:
            writer = DataclassWriter(f, keyed_data, CanonData)
            writer.write()

    def create_test_data(self):
        unkeyed_canons = list(self.unkeyed_path.iterdir())
        unkeyed_data = []
        for file in unkeyed_canons:
            unkeyed_data.append(
                CanonData(
                    canon="",
                    file_name=file.name,
                    classification_label="",
                    num_files=-1,
                )
            )

        with open(self.cfg.project_root / self.cfg.test_out_path, "w") as f:
            writer = DataclassWriter(f, unkeyed_data, CanonData)
            writer.write()

    def run(self):
        self.create_train_data()
        self.create_test_data()


if __name__ == "__main__":
    cfg = DotMap(
        project_root=Path("/mounts/u-amo-d1/adibm-data/projects/latin"),
        data_path="data/",
        test_out_path="data/test_unkeyed_data.csv",
        train_out_path="data/train_unkeyed_data.csv",
        data_type="canon",
    )

    cd = CreateUnkeyedData(cfg)
    cd.run()
