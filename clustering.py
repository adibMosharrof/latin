import os
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

from lat_bert import LatinBERT
from dotmap import DotMap
from dataclasses import dataclass


@dataclass
class LatinData:
    file_name: str
    folder_name: str
    text: str


class LatinClustering:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def load_text(self):
        out = []
        all_canons = os.listdir(self.cfg.data_path)[:10]

        for canon_dir in all_canons:
            files = os.listdir(self.cfg.data_path / canon_dir)
            for file in files:
                with open(self.cfg.data_path / canon_dir / file, "r") as f:
                    text = f.read()
                    out.append(LatinData(file, canon_dir, text))
        return out

    def get_encodings(self, data):
        bert = LatinBERT(
            tokenizerPath=self.cfg.tokenizer_path, bertPath=self.cfg.bert_path
        )
        bert_sents=bert.get_berts(sents)
    def run(self):
        # bert = LatinBERT(tokenizerPath=tokenizerPath, bertPath=bertPath)
        data = self.load_text()
        a = 1


# python3 scripts/gen_berts.py --bertPath models/latin_bert/ --tokenizerPath models/subword_tokenizer_latin/latin.subword.encoder
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-b', '--bertPath', help='path to pre-trained BERT', required=True)
    # parser.add_argument('-t', '--tokenizerPath', help='path to Latin WordPiece tokenizer', required=True)

    # args = vars(parser.parse_args())

    # bertPath=args["bertPath"]
    # tokenizerPath=args["tokenizerPath"]

    # bert=LatinBERT(tokenizerPath=tokenizerPath, bertPath=bertPath)

    # sents=["arma virumque cano", "arma gravi numero violentaque bella parabam"]

    # bert_sents=bert.get_berts(sents)

    # for sent in bert_sents:
    # 	for (token, bert) in sent:
    # 		print("%s\t%s" % ( token, ' '.join(["%.5f" % x for x in bert])))
    # 	print()
    cfg = DotMap(
        data_path=Path("data/keyed/canon"),
        bert_path=Path("models/latin_bert"),
        tokenizer_path=Path("models/latin_bert/latin.subword.encoder"),
    )

    lc = LatinClustering(cfg)
    lc.run()
