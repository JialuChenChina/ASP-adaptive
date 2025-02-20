<img width="948" alt="image" src="https://github.com/user-attachments/assets/7ca550e1-9b8f-4be4-a138-d79827d6ad7d" /># ASP-adaptive

This repo is Pytorch implemention of TNNLS as described in the paper: “Unifying Attribute and Structure Preservation for Enhanced Graph Contrastive Learning”.

## Files
```
   .
    ├── code                           # Codes
    │   ├── eval.py                    # The toolkits for evaluation for node classification task.
    │   ├── evaluate.py                # The toolkits for evaluation for link prediction task.
    │   ├── model.py                   # Code for building up model.
    │   ├── run.sh                     # Reproduce the results reported in our paper.
    │   ├── train.py                   # Training process for node classification.
    │   ├── train_link.py              # Training process for link prediction.
    │   └── utils.py                   
    │
    ├── data                           # Datasets
    └── knn                            # Generated KNN graph
```
## Usage
**Command for node classification task on Cora dataset**
```bash
python train.py \
    --dataset=cora  \
    --SEED=0 \
    --K=1 \
    --lr_p=0.001 \
    --wd_p=0.0001 \
    --num_epochs=500 \
    --num_hidden=256 \
    --tau1=1.2 \
    --l1=1.0 \
    --metric='jaccard' \
    --optimizer='adam' \
    --drop_edge_rate=0.1 \
    --drop_kg_edge_rate=0.2
```
**Command for link prediction task on Cora dataset**
```bash
python train_link.py \
    --dataset=cora  \
    --SEED=0 \
    --K=100 \
    --lr_p=0.001 \
    --wd_p=0.0001 \
    --num_epochs=500 \
    --num_hidden=256 \
    --tau1=0.2 \
    --l1=1.0 \
    --metric='jaccard' \
    --optimizer='adam' \
    --drop_edge_rate=0.1 \
    --drop_kg_edge_rate=0.2 \
    --task='link'
```
