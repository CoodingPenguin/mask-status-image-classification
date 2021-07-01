## 📁 File Structure

### Code Folder

```text
code/   # model training code folder
│
├── dataset.py          # dataset class
├── evaluation.py       # calculate accuracy and f1 score
├── inference.py        # inference
├── loss.py             # loss definition
├── madgrad.py          # optimizer madgrad code
├── model.py            # model definition
├── train.py            # train
└── ensemble.ipynb      # ensemble with soft voting
```

### Dataset Folder

```text
input/data/ # dataset folder
│
├── train/      # train dataset
│   ├── images/     # image folder
│   │   ├── 0000001_female_Asian_45
│   │   │   ├── incorrect_mask.jpg
│   │   │   ├── mask1.jpg
│   │   │   ├── mask2.jpg
│   │   │   ├── mask3.jpg
│   │   │   ├── mask4.jpg
│   │   │   ├── mask5.jpg
│   │   │   └── normal.jpg
│   │   ├── 000002_female_Asian_52
│   │   └── ...
│   └── train.csv   # data metadata file (id, gender, race, age, path)
│
└── eval/   # test dataset
    ├── images/     # image folder
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    └── info.csv    # submission file (id, class)

```

## 🛠 Installation

```shell
pip install -r requirements.txt
```

- torch==1.6.0
- torchvision==0.7.0
- tensorboard==2.4.1
- pandas==1.1.5
- opencv-python==4.5.1.48
- scikit-learn~=0.24.1
- matplotlib==3.2.1

## ⏩ Usage

### Train

```shell
SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py [--seed] [--epochs]\
[--dataset] [--augmentation] [--resize] [--batch_size] [--valid_batch_size] [--model] [--optimizer] [--lr]\
[--val_ratio] [--criterion] [--lr_decay_step] [--log_interval] [--name]
```

### Inference

```shell
SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir]\
python inference.py [--batch_size] [--resize] [--model] [--model_dir] [--output_dir]
```

### Evaluation

```shell
SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py
```
