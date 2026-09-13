# JFER

Anonymous implementation of Joint Future Exploration and Reasoning for
interactive motion forecasting.

JFER contains structured joint-hypothesis construction, complementary
joint-future expansion, candidate-aware refinement, and
reliability-constrained consolidation.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Set `DATA_ROOT` and the pretrained joint forecaster path `PRETRAINED_MODEL` in
`config.yaml`.

## Training

```bash
python train.py
```

The training entry point runs all JFER stages in sequence. Individual stages
can be selected with `--stage`.

## Evaluation

```bash
python eval.py --checkpoint /path/to/checkpoint.pth
```

The method implementation is organized under `models/jfer/`; the scene
encoder, base decoder, and CUDA operators are included in this repository.
