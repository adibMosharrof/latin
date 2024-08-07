from pathlib import Path
from dotmap import DotMap
import pandas as pd
from glob import glob


class DataStats:
    def __init__(self, cfg):
        self.cfg = cfg

    def file_names_df(self):
        extension = ".txt"
        file_names = glob(
            str(self.cfg.project_root / self.cfg.data_path) + f"/**/*{extension}",
            recursive=True,
        )
        data = []
        for file in file_names:
            p = Path(file)
            data.append(
                {
                    "canon": p.parent.name,
                    "file_name": p.stem,
                    "file_path": file,
                }
            )
        return pd.DataFrame(data)

    def run(self):
        df = self.file_names_df()
        df_group_canon = df.groupby("canon")
        all_data_stats = []
        filtered_canons = []
        for name, group in df_group_canon:
            all_data_stats.append(
                {
                    "canon": name,
                    "num_files": group.shape[0],
                }
            )
            if group.shape[0] > 3:
                filtered_canons.append(name)

        pd.DataFrame(all_data_stats).to_csv(
            self.cfg.project_root / "data" / "data_stats" / self.cfg.stats_out_path,
            index=False,
        )
        pd.DataFrame(filtered_canons, columns=["canon"]).to_csv(
            self.cfg.project_root / "data" / "data_stats" / self.cfg.filtered_csvs,
            index=False,
        )


if __name__ == "__main__":
    cfg = DotMap(
        project_root=Path("/mounts/u-amo-d1/adibm-data/projects/latin"),
        # data_path="data/keyed/canon/",
        data_path="data/keyed/decrExcerpt/",
        stats_out_path="decr_data_stats.csv",
        filtered_csvs="filtered_decrs.csv",
    )

    ds = DataStats(cfg)
    ds.run()
