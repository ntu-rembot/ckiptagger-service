# CKIPTagger Service

Serve [ckiplab/ckiptagger](https://github.com/ckiplab/ckiptagger) with [bentoml](https://github.com/bentoml/BentoML)

## Getting started
```bash
poetry install
poetry shell

# download model files
python -c "from ckiptagger import data_utils; data_utils.download_data_gdown('./')"

# export model to bentoml model store
python script.py
```

Option 1: Direct serve
```bash
bentoml serve service:CKIPTaggerService
```

Option 2: Containerize
```bash
bentoml build
bentoml containerize ckip_tagger_service:latest

docker run --rm -p 3000:3000 ckip_tagger_service:<tag>
```
