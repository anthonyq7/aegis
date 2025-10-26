import torch
from torch.utils.data import Dataset
import pandas as pd

#PyTorch's Dataset is base class for custom dataset
#A dataset is a blueprint for how to unload and organize data -> a smart container that knows how to load, process, and return data in correct format for the model
#torch is needed to create tensors
class CodeDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=512):

        self.data = pd.read_json(filepath, lines=True)

        self.tokenizer = tokenizer

        self.max_length = max_length

        print(f"Loaded {len(self.data)} samples from {filepath}")
    
    #Required by PyTorch to know how many samples exist
    #Python magic method so you can do len(dataset) to find length
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]

        #tokenizes the sample with padding and truncation
        encoding = self.tokenizer(
            row["code"], #actual code string
            truncation = True, #truncates if bigger than max length
            padding = "max_length", #pads (adds token) if < max_length to reach 512 tokens
            max_length = self.max_length, #the target length: 512 tokens
            return_tensors = "pt" #Return PyTorch tensors
        )

        #These specific keys are REQUIRED for most HuggingFace models like BERT
        #.flatten converst multi-dimensional array into a one-dimensional array
        return {
            "input_ids": encoding["input_ids"].flatten(), #The tokenized code
            "attention_mask": encoding["attention_mask"].flatten(), #Tells model which tokens are real vs padding
            "labels": torch.tensor(row["label"], dtype=torch.long) #The ground truth (0 or 1)
        }
