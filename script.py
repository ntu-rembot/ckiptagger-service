"""Execute this script to save the models to the BentoML model store."""

import shutil
from pathlib import Path

import bentoml

base_dir = Path.cwd()

models = base_dir.joinpath("data")
with bentoml.models.create(name="ckiptagger") as model_ref:
    shutil.copytree(models, model_ref.path, dirs_exist_ok=True)
    print(f"Model saved: {model_ref}")
