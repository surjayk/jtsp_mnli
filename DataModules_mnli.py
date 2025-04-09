# DataModules_mnli.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from Constants_mnli import hyperparameters

SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'

class MNLIJointDataset(Dataset):
    def __init__(self, hf_split, main_tokenizer=None, policy_tokenizer=None, max_length=128, device="cpu"):
        self.hf_split = hf_split
        self.main_tokenizer = main_tokenizer if main_tokenizer is not None else AutoTokenizer.from_pretrained(hyperparameters["model_name"])
        self.policy_tokenizer = policy_tokenizer if policy_tokenizer is not None else AutoTokenizer.from_pretrained(hyperparameters["defer_model_name"])
        self.max_length = max_length
        self.device = device
        self.data = []
        
        for row in hf_split:
            text1 = row["text1"]
            text2 = row["text2"]
            label = row["label"]
            #skip examples with label -1 (unlabeled test split)
            if label == -1:
                continue
            #create an embedding as the concatenation of text1 and text2 with CLS and SEP tokens
            embedding = CLS_TOKEN + " " + text1 + " " + SEP_TOKEN + " " + text2
            data_list = [embedding, embedding]
            self.data.append({
                "label": label,
                "embedding": embedding,
                "data_list": data_list,
                "text1": text1,
                "text2": text2,
            })
        self.tag2id = {0: 0, 1: 1, 2: 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        label = item["label"]
        embedding = item["embedding"]
        data_list = item["data_list"]

        input_ids = []
        attention_masks = []
        for line in data_list:
            tokenized = self.main_tokenizer(
                line,
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            input_ids.append(tokenized["input_ids"])
            attention_masks.append(tokenized["attention_mask"])
        
        tokenized_emb = self.policy_tokenizer(
            embedding,
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        emb_ids = tokenized_emb["input_ids"]
        emb_attention_mask = tokenized_emb["attention_mask"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=self.device),
            "embedding": torch.tensor(emb_ids, dtype=torch.long, device=self.device),
            "emb_attention_mask": torch.tensor(emb_attention_mask, dtype=torch.long, device=self.device),
            "label": torch.tensor(label, dtype=torch.long, device=self.device),
        }

def load_mnli_splits():
    dataset = load_dataset("SetFit/mnli")
    full_train = dataset["train"]       # ~393k examples
    val_split = dataset["validation"]     # ~9.8k examples
    test_split = dataset["test"]          # ~9.8k examples, labels = -1
    return full_train, val_split, test_split

def create_dataloaders(device="cpu"):
    full_train, val_raw, test_raw = load_mnli_splits()

    SUBSET_SIZE = hyperparameters["SUBSET_SIZE"]
    TRAIN_SIZE = hyperparameters["TRAIN_SIZE"]
    TEST_SIZE = hyperparameters["TEST_SIZE"]
    VAL_SIZE = hyperparameters["VAL_SIZE"]

    small_train = full_train.select(range(SUBSET_SIZE))
    train_subset = small_train.select(range(TRAIN_SIZE))
    test_subset = small_train.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    val_subset = val_raw.select(range(VAL_SIZE))

    main_tokenizer = AutoTokenizer.from_pretrained(hyperparameters["model_name"])
    policy_tokenizer = AutoTokenizer.from_pretrained(hyperparameters["defer_model_name"])

    train_dataset = MNLIJointDataset(train_subset, main_tokenizer, policy_tokenizer, max_length=hyperparameters["max_length"], device=device)
    val_dataset   = MNLIJointDataset(val_subset, main_tokenizer, policy_tokenizer, max_length=hyperparameters["max_length"], device=device)
    test_dataset  = MNLIJointDataset(test_subset, main_tokenizer, policy_tokenizer, max_length=hyperparameters["max_length"], device=device)

    train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=hyperparameters["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=hyperparameters["batch_size"], shuffle=False)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader
