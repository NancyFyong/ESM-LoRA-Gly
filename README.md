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

### Quick predict glycosylation sites of N-GlycositeAltas dataset and O-linked dataset:
put checkpoints under folder `checkpoints/`:

    ├── checkpoints
    │       └── N-linked
    │             └──ESM-3B
    |             └──ESM-150M
    │       └── O-linked
    │             └──ESM-3B
    ├── data
    ├── model
    ├── scripts
    ├── main.py
    ├── predict.py



### Train the classifier of ESM-LoRA-Gly on N-GlycositeAltas dataset:
```
bash scripts/train.sh
```

### eval  N-linked glycosylation sites of N-GlycositeAltas dataset（sigle site）:
```
bash scritps/test.sh
```

### predict  N-linked glycosylation sites(sigle protein):
```
python inference.py
```

