[README.md](https://github.com/user-attachments/files/32156946/README.md)
# JFER

Anonymous implementation for the submitted paper.


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
