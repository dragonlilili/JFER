[README.md](https://github.com/user-attachments/files/32156946/README.md)
# JFER

Anonymous implementation for the submitted paper.

JFER produces six scene-consistent joint futures through four components:

1. Structured Joint-Hypothesis Construction
2. Complementary Joint-Future Expansion
3. Candidate-Aware Refinement
4. Reliability-Constrained Consolidation

The repository contains the JFER extension and uses MTR as an external scene
encoder and motion-decoding backbone. The MTR source tree and WOMD data are not
included.

The implementation is organized by method responsibility:

```text
models/modules/
├── joint_hypothesis_pipeline.py
├── joint_reasoning.py
├── candidate_refinement.py
├── spatial_context.py
├── horizon_scoring.py
└── jfer_decoder.py
```

Backbone integration, WOMD data handling, evaluation, and training utilities
are kept in `models/backbone`, `data`, `evaluation`, and `utils`, respectively.

## Environment

Install the official MTR repository and compile its CUDA extensions. Then
install the additional Python dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Follow the official WOMD preprocessing procedure used by MTR. Set `MTR_ROOT`
and `DATA_ROOT` in `config.yaml` before running the commands below.

The expected dataset files are:

```text
processed_scenarios_training/
processed_scenarios_validation_interactive/
processed_scenarios_training_official_ooi_infos.pkl
processed_scenarios_validation_interactive_infos.pkl
```

## Training

```bash
python train.py
```

The released protocol keeps the converged Base predictor fixed and optimizes
the candidate-aware residual refinement stage. `train.py` installs the minimal
JFER extension into the configured MTR checkout and starts distributed
training. The included checkpoint is used as the default initialization; a
different compatible initialization can be supplied with `--checkpoint`.

## Evaluation

```bash
python eval.py
```

To evaluate another checkpoint:

```bash
python eval.py --checkpoint /path/to/checkpoint.pth
```

The included `best_model.pth` has SHA256:

```text
6efae4a3275bea8effb44795be6ec16cdd50a3163ea2494dd863bbdf60d641f4
```

The checkpoint is tracked with Git LFS because it exceeds GitHub's standard
file-size limit. Run `git lfs install` before the first commit.

## Reference Validation Result

The released checkpoint was evaluated on 43,479 WOMD interaction validation
scenes.

| Soft-mAP | mAP | Miss Rate | minADE | minFDE | Overlap Rate |
|---:|---:|---:|---:|---:|---:|
| 0.286808 | 0.279895 | 0.414393 | 0.864053 | 1.915199 | 0.055083 |
