# DataModules_mnli.py

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from Constants_mnli import hyperparameters

class MNLIDataset(Dataset):
    def __init__(self, hf_split, tokenizer=None, max_length=None):
        self.hf_split = hf_split
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(hyperparameters["model_name"])
        self.policy_tokenizer = AutoTokenizer.from_pretrained(hyperparameters["defer_model_name"])  # Deferral model tokenizer
        self.max_length = max_length if max_length else hyperparameters["max_length"]

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        row = self.hf_split[idx]
        text1 = row["text1"]
        text2 = row["text2"]
        label = row["label"]  # 0, 1, 2 or -1

        # Tokenization for the classifier
        encodings = self.tokenizer(
            text1,
            text2,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        embedding_str = "[CLS] " + text1 + " [SEP] " + text2  

        policy_enc = self.policy_tokenizer(
            embedding_str,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        embedding_ids = policy_enc["input_ids"].squeeze(0)
        embedding_mask = policy_enc["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "embedding_ids": embedding_ids,     # For deferral policy
            "embedding_mask": embedding_mask,   # For deferral policy
            "label": label
        }


def load_mnli_splits():
    dataset = load_dataset("SetFit/mnli")
    # Official splits: train (393k), validation (9.8k), test (9.8k) with label=-1
    train_split = dataset["train"]
    val_split   = dataset["validation"]
    test_split  = dataset["test"]
    return train_split, val_split, test_split


def create_dataloaders():
    # Load raw splits from Hugging Face
    hf_train, hf_val, hf_test = load_mnli_splits()

    # Grab subset hyperparameters
    SUBSET_SIZE = hyperparameters["SUBSET_SIZE"]
    TRAIN_SIZE  = hyperparameters["TRAIN_SIZE"]
    TEST_SIZE   = hyperparameters["TEST_SIZE"]
    VAL_SIZE    = hyperparameters["VAL_SIZE"]

    # Subset train split
    if SUBSET_SIZE < len(hf_train):
        hf_train = hf_train.select(range(SUBSET_SIZE))

    # Split into train and internal test set
    train_subset = hf_train.select(range(TRAIN_SIZE))
    test_subset  = hf_train.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    # Select validation subset
    val_subset = hf_val.select(range(VAL_SIZE)) if VAL_SIZE < len(hf_val) else hf_val

    # Tokenizer initialization
    tokenizer = AutoTokenizer.from_pretrained(hyperparameters["model_name"])

    # Build datasets
    train_dataset = MNLIDataset(train_subset, tokenizer)
    val_dataset   = MNLIDataset(val_subset, tokenizer)
    test_dataset  = MNLIDataset(test_subset, tokenizer)

    # Build dataloaders
    batch_size = hyperparameters["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_dataset)}; Val size: {len(val_dataset)}; Test size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader
