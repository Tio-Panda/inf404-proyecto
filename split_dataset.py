import math
import os
import shutil
import sys
import tarfile
from pathlib import Path
import random

from huggingface_hub import snapshot_download


def download_dataset(cache_path: Path):
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset_path = snapshot_download(
        repo_id="neuroback/DataBack", repo_type="dataset", cache_dir=cache_path
    )
    dataset_path = Path(dataset_path)
    return dataset_path


def decompress_dataset(
    full_data_path: Path, target_path: Path, original=True, dual=True
):
    datasets = []
    if original:
        datasets.append("original")
    if dual:
        datasets.append("dual")

    if len(datasets) == 0:
        return "no selected dataset"

    for dataset_name in datasets:
        search_dir = full_data_path / dataset_name
        targz_files = list(search_dir.glob("*.tar.gz"))

        for file in targz_files:
            print(f"Sorting {file.name}...")

            try:
                tarfile.open(file, "r:gz").extractall(path=target_path)
            except tarfile.ReadError:
                print(f"Could not extract {file.name}", file=sys.stderr)
                continue


def get_source_list(original=True, dual=True, finetune=True):
    pt_sources = []
    ft_sources = []

    if original:
        pt_sources += [("cnf_pt", "bb_pt")]
        if finetune:
            ft_sources += [("cnf_ft", "bb_ft")]

    if dual:
        pt_sources += [("d_cnf_pt", "d_bb_pt")]
        if finetune:
            ft_sources += [("d_cnf_ft", "d_bb_ft")]

    return pt_sources, ft_sources

def get_filtered_cnf_list(data_path, filters, sources, n=-1, shuffle=True, seed=11):
    kb = filters["KB"]

    data = []

    for source in sources:
        cnf_source = source[0]
        bb_source = source[1]

        cnf_path = data_path / cnf_source
        cnf = [
            file
            for file in cnf_path.iterdir()
            if file.is_file() and file.stat().st_size / 1024 <= kb
        ]
        
        bb_path = data_path / bb_source
        bb = [
            Path(f"{bb_path.resolve()}/{file.stem}.backbone.xz")
            for file in cnf
        ]

        _data = [
            {"cnf": a, "bb": b}
            for a, b in zip(cnf, bb)
        ]

        data += _data

    if n >= 0:
        data = data[:n]

    if shuffle:
        random.seed(seed)
        random.shuffle(data)

    return data

def split_dataset(files, target_data, ratios=[0.7, 0.2, 0.1]):
    
    if target_data.exists():
        shutil.rmtree(target_data)

    target_data.mkdir(parents=True, exist_ok=True)

    train_r, val_r, _ = ratios
    n = len(files)

    train_n = math.floor(n * train_r)
    val_n = math.floor(n * val_r)

    train = files[:train_n]
    val = files[train_n:train_n + val_n]
    test = files[train_n + val_n:]


    dataset_name = ["pretrain", "validation", "test"]
    sources = ["cnf", "backbone"]

    for d_name in dataset_name:
        for source in sources:
            _path = target_data / source / d_name
            _path.mkdir(parents=True, exist_ok=True)

    datasets = [("pretrain", train), ("validation", val), ("test", test)]

    for d_name, split_data in datasets:
        for data in split_data:
            cnf = data["cnf"]
            bb = data["bb"]

            cnf_sym_path = target_data / sources[0] / d_name / cnf.name
            cnf_path = cnf.resolve()
            os.symlink(cnf_path, cnf_sym_path)

            bb_sym_path = target_data / sources[1] / d_name / bb.name
            bb_path = bb.resolve()
            os.symlink(bb_path, bb_sym_path)

if __name__ == "__main__":
    FULL_DATA_PATH = Path("./full_data")
    PROCESSED_DATA_PATH = Path("./processed_data/")
    SPLIT_DATA_PATH = Path("./sym_data")

    TRAIN_RATIO = 0.7
    VALIDATION_RATIO = 0.2
    TEST_RATIO = 0.1

    data_path = download_dataset(FULL_DATA_PATH)
    
    decompress_dataset(data_path, PROCESSED_DATA_PATH)

    pt_sources, ft_sources = get_source_list()
    sources = pt_sources + ft_sources
    
    filters = {"KB": 6}
    n = 5000
    data = get_filtered_cnf_list(PROCESSED_DATA_PATH, filters, sources, n=n, shuffle=True, seed=13)

    print("n dataset:", len(data))

    ratios = [TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO]
    split_dataset(data, SPLIT_DATA_PATH, ratios)
