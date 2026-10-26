"""Download everything the experiments read into data/: RefCOCO/+/g annotations, COCO-2017 train images, and the Jag
release (baseline weights).

    python scripts/setup_data.py [--no-coco] [--no-jag]

RefCOCO/+/g: jxu124/{refcoco,refcocoplus,refcocog} parquet on the Hugging Face Hub (UNC / UMD splits; bbox = xyxy
pixels) -> data/refcoco/<name>/{train,validation,test[,testB]}.parquet. COCO train2014 ids are the same files as
train2017, so only train2017 is needed (pcuenq/coco-2017-mirror, ~18 GB). Jag: xyzzzh/Jag -> data/external/jag_repo.
"""
import argparse
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

DATA = Path(__file__).resolve().parents[1] / "data"


def refcoco():
    for name in ("refcoco", "refcocoplus", "refcocog"):
        dst = DATA / "refcoco" / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in list_repo_files(f"jxu124/{name}", repo_type="dataset"):
            if f.endswith(".parquet"):
                split = Path(f).name.split("-")[0]  # train / validation / test / testB
                shutil.copy(hf_hub_download(f"jxu124/{name}", f, repo_type="dataset"), dst / f"{split}.parquet")
        print(name, sorted(p.name for p in dst.iterdir()))


def coco():
    dst = DATA / "coco"
    if (dst / "train2017").is_dir():
        print("coco: train2017 present")
        return
    z = hf_hub_download("pcuenq/coco-2017-mirror", "train2017.zip", repo_type="dataset")
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(dst)
    print("coco:", sum(1 for _ in (dst / "train2017").iterdir()), "images")


def jag():
    print("jag:", snapshot_download("xyzzzh/Jag", local_dir=DATA / "external" / "jag_repo"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-coco", action="store_true")
    ap.add_argument("--no-jag", action="store_true")
    a = ap.parse_args()
    refcoco()
    if not a.no_coco:
        coco()
    if not a.no_jag:
        jag()
