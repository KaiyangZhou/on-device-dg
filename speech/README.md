# Speech Recognition

This code can be used to reproduce the experiments on [Google Speech Commands](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html).

## Instructions

### 1. Dataset

To download the dataset, follow these steps:
```
# Create a data folder
mkdir data

# Download the dataset (this takes some time so sit back and relax)
python download.py

# The output from your screen should be:
# Train: 84,843
# Val: 9,981
# Test: 11,005
# Done!
```

In the end, you should have `data/SpeechCommands`.

To reproduce the results, you should download the three random splits used in the paper from this [link](https://drive.google.com/file/d/1U40l66LU1VVksq9dHnPu2gwpHXZfK8ZQ/view?usp=share_link) and unzip the folder under `data/`.

The final file structure should look like
```
data/
    SpeechCommands/
    zhou_splits/
```

Then you are ready to go.

### 2. Training

The training command for each model is shown below.

- **ERM (M11, the teacher model)**: `python main.py -i 1 -m M11 --alg erm -o output/erm/M11/seed1`
- **ERM (M3)**: `python main.py -i 1 -m M3 --alg erm -o output/erm/M3/seed1`
- **KD (M3)**: `python main.py -i 1 -m M3 --alg kd -o output/kd/M3/seed1`
- **OKD (M3, mixup)**: `python main.py -i 1 -m M3 --alg okd --ood_type mixup -o output/okd/mixup/M3/seed1`
- **OKD (M3, mask)**: `python main.py -i 1 -m M3 --alg okd --ood_type mask -o output/okd/mask/M3/seed1`
- **OKD (M3, noise)**: `python main.py -i 1 -m M3 --alg okd --ood_type noise -o output/okd/noise/M3/seed1`

For KD/OKD, you need to train the teacher model first, i.e., ERM (M11).

Each model should be run three times for the three splits (by varying `-i`). For example, for ERM (M3), you should run
```sh
python main.py -i 1 -m M3 --alg erm -o output/erm/M3/seed1
python main.py -i 2 -m M3 --alg erm -o output/erm/M3/seed2
python main.py -i 3 -m M3 --alg erm -o output/erm/M3/seed3
```
