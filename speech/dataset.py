import os
import sys
import json
import random
from tqdm import tqdm
from collections import defaultdict

import torch
from torchaudio.datasets import SPEECHCOMMANDS


CLASSES = [
    "backward", "bed", "bird", "cat", "dog", "down",
    "eight", "five", "follow", "forward", "four",
    "go", "happy", "house", "learn", "left", "marvin",
    "nine", "no", "off", "on", "one", "right", "seven",
    "sheila", "six", "stop", "three", "tree", "two",
    "up", "visual", "wow", "yes", "zero"
]


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./data", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


class SubsetSC2(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, split_id: int = 1):
        super().__init__("./data", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
        
        self._walker = load_list(f"../../zhou_splits/zhou_speechcommands_v{split_id}_{subset}.txt")


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(CLASSES.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return CLASSES[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def save_json(obj, fp):
    with open(fp, "w") as f:
        json.dump(obj, f, indent=4)


def create_new_splits(switch="off"):
    # waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

    print(f"Switch: {switch}")
    if switch == "off":
        sys.exit()
    
    train_set = SubsetSC("training")
    val_set = SubsetSC("validation")
    test_set = SubsetSC("testing")
    all_set = train_set + val_set + test_set

    speakers_data = defaultdict(list)
    for i, audio in enumerate(all_set):
        sid = audio[3]
        speakers_data[sid].append(i)
    
    all_id_list = sorted(list(speakers_data.keys()))
    n_speaker = len(all_id_list)
    n_test = int(n_speaker * 0.2)
    n_val = int(n_speaker * 0.1)

    print(f"Total audio data: {len(all_set):,}")
    print(f"Total speakers: {n_speaker:,}")

    def _write_split(k, split_name, id_list):
        with open(f"data/zhou_speechcommands_v{k}_{split_name}.txt", "w") as f:
            for sid in id_list:
                idxs = speakers_data[sid]
                for idx in idxs:
                    audio = all_set[idx]
                    label = audio[2]
                    utnum = audio[4]
                    line = label + "/" + sid + "_nohash_" + str(utnum) + ".wav\n"
                    f.write(line)

    for k in range(3):
        print(f"Creating split-{k + 1} ...")
        
        test_id_list = random.sample(all_id_list, n_test)
        remains = [sid for sid in all_id_list if sid not in test_id_list]
        val_id_list = random.sample(remains, n_val)
        train_id_list = [sid for sid in remains if sid not in val_id_list]

        _write_split(k + 1, "train", train_id_list)
        _write_split(k + 1, "val", val_id_list)
        _write_split(k + 1, "test", test_id_list)


def build_loaders(split_id=1, batch_size=256, preload=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
    
    train_set = SubsetSC2("train", split_id)
    val_set = SubsetSC2("val", split_id)
    test_set = SubsetSC2("test", split_id)

    if preload:
        print("Preloading data")
        train_set = [item for item in tqdm(train_set)]
        val_set = [item for item in tqdm(val_set)]
        test_set = [item for item in tqdm(test_set)]
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
