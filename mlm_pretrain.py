from cltk.data.fetch import FetchCorpus
import os
from pathlib import Path

cd = FetchCorpus("lat")

a = cd.import_corpus("lat_text_tesserae")
b = 1
