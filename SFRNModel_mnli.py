# SFRNModel_mnli.py
import torch
import torch.nn as nn
from transformers import AutoModel
from Constants_mnli import hyperparameters

class SFRNModel(nn.Module):
    def __init__(self, num_labels):
        super(SFRNModel, self).__init__()
        self.bert = AutoModel.from_pretrained(hyperparameters['model_name'])
        self.dropout = nn.Dropout(hyperparameters['hidden_dropout_prob'])

        bert_hidden_size = self.bert.config.hidden_size  # typically 768
        mlp_hidden = hyperparameters['mlp_hidden']

        self.g = nn.Sequential(
            nn.Linear(bert_hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        self.f = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, num_labels),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # shape [batch, 768]
        pooled_output = self.dropout(pooled_output)

        hidden = self.g(pooled_output)      # shape [batch, mlp_hidden]
        logits = self.f(hidden)            # shape [batch, num_labels]

        return logits, hidden


class DeferralClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=2):
        super(DeferralClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, embedding_ids, hidden_states, attention_mask=None):
        temp = self.fc1(hidden_states)  
        out = self.fc2(temp)
        return out
