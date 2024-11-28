import tarfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

from MapLocNet import logger


def download(data_dir: Path):
    data_dir.mkdir(exist_ok=True, parents=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    keys = [
        "public/v1.0/v1.0-test_blobs_camera.tgz",
        "public/v1.0/v1.0-test_meta.tgz",
        "public/v1.0/v1.0-trainval01_blobs_camera.tgz",
        "public/v1.0/v1.0-trainval02_blobs_camera.tgz",
        "public/v1.0/v1.0-trainval03_blobs_camera.tgz",
        "public/v1.0/v1.0-trainval04_blobs_camera.tgz",
        "public/v1.0/v1.0-trainval05_blobs_camera.tgz",
        "public/v1.0/v1.0-trainval06_blobs_camera.tgz",
        "public/v1.0/v1.0-trainval07_blobs_camera.tgz",
        "public/v1.0/v1.0-trainval08_blobs_camera.tgz",
        "public/v1.0/v1.0-trainval09_blobs_camera.tgz",
        "public/v1.0/v1.0-trainval10_blobs_camera.tgz",
        "public/v1.0/v1.0-trainval_meta.tgz",
    ]

    for key in keys:
        local_path = data_dir / key.split("/")[-1]
        if local_path.exists():
            logger.info(f"{local_path} already exists. Skipping...")
            continue

        obj = s3.get_object(Bucket="motional-nuscenes", Key=key)
        file_size = obj["ContentLength"]

        logger.info(f"Downloading {key} to {local_path}...")
        with tqdm(
            total=file_size, unit="B", unit_scale=True, desc=key.split("/")[-1]
        ) as pbar:
            with open(local_path, "wb") as f:
                s3.download_fileobj(
                    Bucket="motional-nuscenes",
                    Key=key,
                    Fileobj=f,
                    Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
                )
        logger.info(f"Downloaded {key} to {local_path}")


def extract(data_dir: Path):
    for file in data_dir.glob("*.tgz"):
        logger.info(f"Extracting {file} to {data_dir}...")
        with tarfile.open(file, "r:gz") as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), unit="files", desc=file.name) as pbar:
                for member in members:
                    tar.extract(member, data_dir)
                    pbar.update(1)
                    pbar.refresh()
        logger.info(f"Extracted {file} to {data_dir}.")


if __name__ == "__main__":
    local_path = Path("datasets/nuScenes")
    logger.info("Downloading nuScenes dataset...")
    download(Path(local_path))
    logger.info("Downloaded nuScenes dataset.")

    logger.info("Extracting nuScenes dataset...")
    extract(local_path)
    logger.info("Extracted nuScenes dataset.")
