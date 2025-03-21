from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_data():
    tarball_path = Path('C:/ML Personal Archive/dataset/housing.tgz')
    if not tarball_path.is_file():
        Path("dataset").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as tarball:
        tarball.extractall(path="dataset")
    return pd.read_csv("dataset/housing/housing.csv")
