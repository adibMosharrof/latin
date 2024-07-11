
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
@dataclass
class CanonData:
    canon:str
    file_name:str
    classification_label:str
    num_files:int



class CreateData:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.data_path = self.cfg.project_root / self.cfg.data_path

    def get_relative_to_data_path(self, path):
        return path.relative_to(self.cfg.data_path)
    
    def get_splits_from_canon_3(self, data):
        filtered = data[data.num_files > 3]
        # final_data = filtered.assign(classification_label=filtered.canon).drop(columns=["num_files"])
        canon_data = []
        for i,item in filtered.iterrows():
            files = list((self.cfg.data_path / item.canon).iterdir())
            for file in files:
                canon_data.append(CanonData(canon=item.canon, file_name=file.name, classification_label=item.canon,num_files=item.num_files))
             
        train, test = train_test_split(canon_data, test_size=0.2)
        return train, test


    def run(self):
        data_csv = pd.read_csv(self.cfg.project_root / self.cfg.data_stats_path)
        single_canon = []
        three_canon_train_data, three_canon_test_data = self.get_splits_from_canon_3(data_csv)
        two_canon_test = []
        two_canon_train = [] 
        canons_less_than_two_df = data_csv[data_csv.num_files < 3]
        for canon in canons_less_than_two_df.canon:
            files = list((self.cfg.data_path / canon).iterdir())
            if len(files) == 1:
                single_canon.append(CanonData(canon=canon, file_name=files[0].name, classification_label="other", num_files=1))
            elif len(files) == 2:
                random.shuffle(files)
                test_data = files.pop()
                two_canon_test.append(
                    CanonData(canon=canon, file_name=test_data.name, classification_label=canon, num_files=2)
                )
                two_canon_train.extend([CanonData(canon=file.parent.name,file_name=file.name, classification_label=file.parent.name,num_files=2) for file in files]) 
        single_train, single_test = train_test_split(single_canon, test_size=0.3)
        all_train_data = single_train+ two_canon_train+ three_canon_train_data
        all_test_data = single_test+ two_canon_test+ three_canon_test_data

        for step_data , path in zip([all_train_data,all_test_data],[self.cfg.train_out_path, self.cfg.test_out_path]):
            with open (self.cfg.project_root / path, "w") as f:
                writer = DataclassWriter(f, step_data, CanonData)
                writer.write()
        a=1



if __name__ == "__main__":
    cfg = DotMap(
        project_root=Path("/mounts/u-amo-d1/adibm-data/projects/latin"),
        data_path="data/keyed/canon/",
        data_stats_path="data/data_stats/data_stats.csv",
        test_out_path="data/test_data.csv",
        train_out_path="data/train_data.csv"
    )

    cd = CreateData(cfg)
    cd.run()