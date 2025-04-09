# SFRNModel_mnli.py

import torch
import torch.nn as nn
from transformers import AutoModel
from Constants_mnli import hyperparameters

class SFRNModel(nn.Module):
    def __init__(self):
        super(SFRNModel, self).__init__()
        self.bert = AutoModel.from_pretrained(hyperparameters['model_name'])
        self.dropout = nn.Dropout(hyperparameters['hidden_dropout_prob'])
        
        num_labels = hyperparameters['num_labels']
        mlp_hidden = hyperparameters['mlp_hidden']
        
        self.g = nn.Sequential(
            nn.Linear(hyperparameters['hidden_dim'], mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )
        
        self.f = nn.Sequential(
            nn.Linear(mlp_hidden, hyperparameters['mlp_hidden']),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hyperparameters['mlp_hidden'], num_labels)
        )
        
        self.alpha = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )
        
        self.beta = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )

    def forward(self, input_ids, emb, attention_mask=None, emb_attention_mask=None):
        main_input_ids = input_ids[:, 0, :]  
        main_attention_mask = attention_mask[:, 0, :] if attention_mask is not None else None
        
        outputs = self.bert(main_input_ids, attention_mask=main_attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        
        pooled_output = self.dropout(pooled_output)
        g_t = self.g(pooled_output)
        g_t = self.alpha(g_t) * g_t + self.beta(g_t)
        output = self.f(g_t)
        logits = torch.softmax(output, dim=1)
        
        return logits, g_t

class DeferralClassifier(nn.Module):
    def __init__(self, output_dim=2):
        super(DeferralClassifier, self).__init__()
        self.defer_layer = AutoModel.from_pretrained(hyperparameters['defer_model_name'])
        self.dropout = nn.Dropout(hyperparameters['hidden_dropout_prob'])
        self.fc1 = nn.Sequential(
            nn.Linear(hyperparameters['p_hidden_dim'], hyperparameters['policy_hidden']),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hyperparameters['policy_hidden'], hyperparameters['policy_hidden'] // 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hyperparameters['policy_hidden'] // 2, output_dim)
        )
    
    def forward(self, x1, x2, attention_mask):
        x = self.defer_layer(x1, attention_mask=attention_mask)
        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            pooled_output = x.pooler_output
        else:
            pooled_output = x.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        temp = self.fc1(pooled_output)
        out = self.fc2(temp)
        return out
