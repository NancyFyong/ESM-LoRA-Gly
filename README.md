# ESM-LoRA-Gly: Improved prediction of N and O-linked glycosylation sites by tuning protein language models with low-rank adaptation (LoRA)

**Zhiyong Feng** <sup>1,2</sup>, Xing Zhang1, He Wang1, Xue Hong1, Jian Zhan1,3, and Yaoqi Zhou<sup>∗</sup>

## Environments
```bash
git clone https://github.com/NancyFyong/ESM-LoRA-Gly.git
conda env create -f env.yml
```

## Datasets

- the N-GlycositeAltas dataset and O-linked dataset under folder `data/`:
    ```
    ├── data
    │   ├── N-GlycositeAltas
    │   │     ├── train.csv
    │   │     ├── test.csv
    │   │     └── valid.csv
    │   ├── O-linked
    │   │     ├── train.csv
    │   │     ├── test.csv
    │   │     └── valid.csv
    ```

## Usage

### Quick predict N-linked glycosylation sites of N-GlycositeAltas dataset:
Download the [checkpoint](https://drive.google.com/drive/folders/1cCCIw5HIgtBylf2oVFgSEaVNE1LGAN4g?usp=sharing) of N-GlyAltas_classifier under folder `checkpoints/`:

    ├── checkpoints
    │       └── N-GlyAltas_classifier.pkl
    ├── data
    ├── log
    ├── model
    ├── scripts
    ├── main.py
    ├── predict.py

```
python predict.py --mode=test_features --data_path=./data/N-GlycositeAltas --ckpt_path=./checkpoints/N-GlyAltas_classifier.pkl 
```

### Train the classifier of EMNgly on N-GlycositeAltas dataset:
```
bash scritps/get_N-GlycositeAltas_train_features.sh
python main.py --mode=train --data_path=./data/N-GlycositeAltas --output_path=./checkpoints/N-GlyAltas_classifier.pkl
```

### Predict  N-linked glycosylation sites of N-GlycositeAltas dataset:
```
bash scritps/get_N-GlycositeAltas_test_features.sh
python predict.py --mode=test --data_path=./data/N-GlycositeAltas --ckpt_path=./checkpoints/N-GlyAltas_classifier.pkl 
```
