# Diffusion with Reinforcement Learning for Pedestrian Trajectory Prediction

## Environment
    Python == 3.8.3
    PyTorch == 1.7.0
    CUDA > 10.1

## Pre-train(MID)

### Step 1: Prepare Data for MID
The preprocessed data splits for the ETH/UCY and SDD are in ```raw_data```. Like MID, we preprocess the data and generate .pkl files for training.

```
python process_data_mid.py
```

The `train/val/test/` splits are the same as those found in [Social GAN]( https://github.com/agrimgupta92/sgan). Please see ```process_data.py``` for detail.

### Step 2: Train MID
 ```
 python main.py --dataset [DATASET]
 ``` 
 Note that ```DATASET``` should from ["eth", "hotel", "univ", "zara1", "zara2", "sdd"]
 
Logs and checkpoints will be automatically saved in experiments/baseline.

## Fine-tune(DRL-for-PTP)

### Step 1: Prepare Data for RL
```
python main.py --process_data True --dataset [DATASET] --eval_at [BEST_EPOCH]
```
```BEST_EPOCH``` refers to the epoch number where the pre-trained model performed the best.

The processed data will be saved in the rl/data folder.

### Step 2: Train DRL for PTP
 
 ```
 python main.py  --dataset [DATASET] --eval_at [BEST_EPOCH]
 ``` 
 

Logs and checkpoints will be automatically saved in rl/log.
