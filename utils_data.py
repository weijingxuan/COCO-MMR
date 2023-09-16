import os
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from utils_prompt import *

import pickle

def save_image_features(image_features, output_path):
    with open(output_path, 'wb') as handle:
        pickle.dump(image_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_image_features(input_path):
    with open(input_path, 'rb') as handle:
        return pickle.load(handle)


img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
}

class COCODatasetImg(Dataset):

    def __init__(
        self, coco_data, qids, tokenizer, source_len, target_len, args, image_features, test_le=None
    ):
        self.tokenizer = tokenizer
        self.data = {d["unique_id"]: d for d in coco_data}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.image_ids = []
        if test_le is not None:
            test_le_data = json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in qids:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair_coco(self.data, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)
            if qid in image_features:
                i_vectors = image_features[qid]
                self.image_ids.append(i_vectors)
            else:
                shape = img_shape[args.img_type]
                self.image_ids.append(np.zeros(shape))
    
    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        image_ids = self.image_ids[index]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        image_ids = torch.tensor(image_ids).squeeze()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids,
        }