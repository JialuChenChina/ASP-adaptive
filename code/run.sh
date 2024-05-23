#!/bin/sh



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
    



python train.py \
    --dataset=citeseer \
    --SEED=0 \
    --K=1 \
    --lr_p=0.0001 \
    --wd_p=0.00001 \
    --num_epochs=500 \
    --num_hidden=256 \
    --tau1=1.5 \
    --l1=0.01 \
    --metric='jaccard'


python train.py \
    --dataset=pubmed  \
    --SEED=0 \
    --K=100 \
    --lr_p=0.001 \
    --wd_p=0.00001 \
    --num_epochs=2000 \
    --num_hidden=256 \
    --tau1=1.0 \
    --l1=100.0 \
    --optimizer='adam' \
    --metric='jaccard' \
    --drop_edge_rate=0.9 \
    
python train.py \
    --dataset=DBLP  \
    --SEED=0 \
    --K=10 \
    --lr_p=0.001 \
    --lr_m=0.01 \
    --wd_p=0.00001 \
    --wd_m=0.0 \
    --num_epochs=1500 \
    --num_hidden=256 \
    --tau1=1.0 \
    --l1=100.0 \
    --optimizer='adam' \
    --metric='jaccard' \
    --drop_edge_rate=0.4 \
    --drop_kg_edge_rate=0.5



python train.py \
    --dataset=Photo  \
    --SEED=0 \
    --K=1 \
    --lr_p=0.001 \
    --wd_p=0.0001 \
    --num_epochs=1000 \
    --num_hidden=256 \
    --tau1=0.8 \
    --l1=100 \
    --optimizer='adam' \
    --drop_edge_rate=0.9 \
    --drop_kg_edge_rate=0.5
    
python train.py \
    --dataset=CS  \
    --SEED=0 \
    --K=100 \
    --lr_p=0.001 \
    --wd_p=0.0001 \
    --num_epochs=1000 \
    --num_hidden=256 \
    --tau1=1.0 \
    --l1=100 \
    --optimizer='adam' \
    --drop_edge_rate=0.9 \
    --drop_kg_edge_rate=0.5
    
    
