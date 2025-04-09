# main_mnli.py

import argparse
import wandb
import random
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
from torch.optim import AdamW
import os

from Constants_mnli import hyperparameters, config_dictionary
from DataModules_mnli import create_dataloaders
from SFRNModel_mnli import SFRNModel, DeferralClassifier

def train(args):
    wandb.init(project="mnli_joint_training", config=config_dictionary)
    random.seed(hyperparameters['random_seed'])
    np.random.seed(hyperparameters['random_seed'])
    torch.manual_seed(hyperparameters['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hyperparameters['random_seed'])
    
    DEVICE = args.device
    print("Using device:", DEVICE)
    print("Model name:", hyperparameters['model_name'])
    print("Hyperparameters:", hyperparameters)
    
    train_loader, val_loader, test_loader = create_dataloaders(device=DEVICE)
    
    model = SFRNModel()
    policy = DeferralClassifier()
    model.to(DEVICE)
    policy.to(DEVICE)
    
    #load pretrained SFRN weights (from pretraining checkpoint)
    pretrained_ckpt = "/data/smk6961/jtsp/SFRN_mnli/checkpoint/checkpoint_sfrn_mnli_pretrain_best.pth"
    if os.path.exists(pretrained_ckpt):
        # Use strict=False to allow missing alpha and beta keys (which will be randomly initialized)
        state_dict = torch.load(pretrained_ckpt, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        print("Loaded pretrained SFRN model from", pretrained_ckpt)
    else:
        print("Pretrained checkpoint not found at", pretrained_ckpt)
    
    optimizer = AdamW(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
    optimizer_p = AdamW(policy.parameters(), lr=hyperparameters['p_lr'], weight_decay=hyperparameters['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    weights = torch.tensor([0.01, 10.0]).to(DEVICE)
    criterion_p = nn.CrossEntropyLoss(weight=weights)
    
    num_training_steps = len(train_loader) * hyperparameters['epochs']
    warmup_steps = int(hyperparameters['WARMUP_STEPS'] * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    scheduler_p = get_linear_schedule_with_warmup(optimizer_p, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    
    grad_accum_steps = hyperparameters['GRADIENT_ACCUMULATION_STEPS']
    
    best_acc, best_f1 = 0, 0
    best_ckp_path = ''
    
    # Training Loop
    for epoch in range(hyperparameters['epochs']):
        if epoch < hyperparameters['pre_step']:
            model.train()
            policy.eval()
        elif epoch < hyperparameters['mid_step']:
            model.eval()
            policy.train()
        else:
            model.train()
            policy.train()
        
        train_loss, train_policy_loss, train_cl_loss = 0.0, 0.0, 0.0
        y_true, y_pred = [], []
        p_true, p_pred = [], []
        y_defer = []
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparameters['epochs']} - Train")
        optimizer.zero_grad()
        optimizer_p.zero_grad()
        for step, batch in enumerate(train_iterator):
            input_ids = batch["input_ids"].to(DEVICE)            # shape: (B, num_candidates, seq_len)
            attention_mask = batch["attention_mask"].to(DEVICE)      # shape: (B, num_candidates, seq_len)
            emb = batch["embedding"].to(DEVICE)                    # shape: (B, seq_len)
            emb_attention_mask = batch["emb_attention_mask"].to(DEVICE)  # shape: (B, seq_len)
            labels = batch["label"].to(DEVICE)                     # shape: (B)
            
            # Forward pass: model returns logits and intermediate features (hidden)
            logits, hidden = model(input_ids, emb, attention_mask=attention_mask, emb_attention_mask=None)
            # Get model predictions (vectorized over batch)
            pred_idx = torch.argmax(logits, dim=1)  # shape: (B)
            # Create deferral labels: 0 if prediction is correct, 1 if incorrect
            defer_label = (pred_idx != labels).long()  # shape: (B)
            
            hidden_unsq = hidden.unsqueeze(1)  # shape: (B, 1, feature_dim)
            defer_logits = policy(emb, hidden_unsq, emb_attention_mask)
            p_pred_idx = torch.argmax(defer_logits, dim=1)  # shape: (B)
            
            loss_cl = criterion(logits, labels)
            loss_defer = criterion_p(defer_logits, defer_label)
            loss_cl = loss_cl / grad_accum_steps
            loss_defer = loss_defer / grad_accum_steps
            
            if epoch < hyperparameters['pre_step']:
                loss = loss_cl
            elif epoch < hyperparameters['mid_step']:
                loss = loss_defer
            else:
                loss = hyperparameters['alpha'] * loss_cl + hyperparameters['beta'] * loss_defer
            
            loss.backward()
            train_loss += loss.item()
            train_cl_loss += loss_cl.item()
            train_policy_loss += loss_defer.item()
            
            # Accumulate metrics over the batch
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred_idx.cpu().numpy().tolist())
            p_true.extend(defer_label.cpu().numpy().tolist())
            p_pred.extend(p_pred_idx.cpu().numpy().tolist())
            # For final prediction, if the deferral policy predicts 1 (defer), use the true label; otherwise, use model prediction
            final_pred = torch.where(p_pred_idx == 1, labels, pred_idx)
            y_defer.extend(final_pred.cpu().numpy().tolist())
            
            if (step + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), hyperparameters['max_norm'])
                nn.utils.clip_grad_norm_(policy.parameters(), hyperparameters['max_norm'])
                if epoch < hyperparameters['pre_step']:
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizer_p.zero_grad()
                    scheduler.step()
                elif epoch < hyperparameters['mid_step']:
                    optimizer_p.step()
                    optimizer.zero_grad()
                    optimizer_p.zero_grad()
                    scheduler_p.step()
                else:
                    optimizer.step()
                    optimizer_p.step()
                    optimizer.zero_grad()
                    optimizer_p.zero_grad()
                    scheduler.step()
                    scheduler_p.step()
                    
        train_f1 = f1_score(y_true, y_pred, average='macro')
        train_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        train_acc = accuracy_score(y_true, y_pred)
        train_policy_acc = accuracy_score(p_true, p_pred)
        train_policy_f1 = f1_score(p_true, p_pred, average='macro')
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, Policy Acc: {train_policy_acc:.4f}")
        
        # Validation Loop
        model.eval()
        policy.eval()
        val_loss, val_policy_loss, val_cl_loss = 0.0, 0.0, 0.0
        val_y_true, val_y_pred = [], []
        val_p_true, val_p_pred = [], []
        val_y_defer = []
        defer_count = 0
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{hyperparameters['epochs']} - Validation")
        for step, batch in enumerate(val_iterator):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            emb = batch["embedding"].to(DEVICE)
            emb_attention_mask = batch["emb_attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            logits, hidden = model(input_ids, emb, attention_mask=attention_mask, emb_attention_mask=None)
            pred_idx = torch.argmax(logits, dim=1)
            defer_label = (pred_idx != labels).long()
            hidden_unsq = hidden.unsqueeze(1)
            defer_logits = policy(emb, hidden_unsq, emb_attention_mask)
            p_pred_idx = torch.argmax(defer_logits, dim=1)
            
            loss_cl = criterion(logits, labels)
            loss_defer = criterion_p(defer_logits, defer_label)
            loss = hyperparameters['alpha'] * loss_cl + hyperparameters['beta'] * loss_defer if epoch > hyperparameters['pre_step'] else loss_cl
            val_loss += loss.item()
            val_cl_loss += loss_cl.item()
            val_policy_loss += loss_defer.item()
            
            val_y_true.extend(labels.cpu().numpy().tolist())
            val_y_pred.extend(pred_idx.cpu().numpy().tolist())
            val_p_true.extend(defer_label.cpu().numpy().tolist())
            val_p_pred.extend(p_pred_idx.cpu().numpy().tolist())
            deferred = torch.where(p_pred_idx == 1, labels, pred_idx)
            val_y_defer.extend(deferred.cpu().numpy().tolist())
            defer_count += (p_pred_idx == 1).sum().item()
            
        val_acc = accuracy_score(val_y_true, val_y_pred)
        val_f1 = f1_score(val_y_true, val_y_pred, average='macro')
        val_qwk = cohen_kappa_score(val_y_true, val_y_pred, weights='quadratic')
        val_policy_acc = accuracy_score(val_p_true, val_p_pred)
        val_policy_f1 = f1_score(val_p_true, val_p_pred, average='macro')
        val_defer_acc = accuracy_score(val_y_true, val_y_defer)
        val_defer_f1 = f1_score(val_y_true, val_y_defer, average='macro')
        val_defer_rate = defer_count / len(val_y_defer) if len(val_y_defer) > 0 else 0
        
        print(f"Epoch {epoch+1}: Validation Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Deferral Rate: {val_defer_rate:.4f}")
        print(f"Train QWK: {train_qwk:.4f}, Val QWK: {val_qwk:.4f}")
        print(f"Train Loss: {train_loss:.4f} (Cl: {train_cl_loss:.4f}, Policy: {train_policy_loss:.4f})")
        
        # Save best model based on validation accuracy and F1
        if (val_acc > best_acc) and (val_f1 > best_f1):
            best_acc = val_acc
            best_f1 = val_f1
            os.makedirs("checkpoint", exist_ok=True)
            best_ckp_path = f"checkpoint/checkpoint_{args.ckp_name}_at_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), best_ckp_path)
            print(f"Best model updated and saved to {best_ckp_path}")
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_cl_loss": train_cl_loss,
            "train_policy_loss": train_policy_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "train_policy_acc": train_policy_acc,
            "train_policy_f1": train_policy_f1,
            "val_loss": val_loss,
            "val_cl_loss": val_cl_loss,
            "val_policy_loss": val_policy_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_policy_acc": val_policy_acc,
            "val_policy_f1": val_policy_f1,
            "val_defer_acc": val_defer_acc,
            "val_defer_f1": val_defer_f1,
            "val_defer_rate": val_defer_rate,
            "train_qwk": train_qwk,
            "val_qwk": val_qwk
        })
        
    # Test Loop
    model.eval()
    policy.eval()
    test_y_true, test_y_pred = [], []
    test_p_true, test_p_pred = [], []
    test_iterator = tqdm(test_loader, desc="Test Iteration")
    for step, batch in enumerate(test_iterator):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        emb = batch["embedding"].to(DEVICE)
        emb_attention_mask = batch["emb_attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        logits, hidden = model(input_ids, emb, attention_mask=attention_mask, emb_attention_mask=None)
        pred_idx = torch.argmax(logits, dim=1)
        defer_label = (pred_idx != labels).long()
        hidden_unsq = hidden.unsqueeze(1)
        defer_logits = policy(emb, hidden_unsq, emb_attention_mask)
        p_pred_idx = torch.argmax(defer_logits, dim=1)
        
        test_y_pred.extend(pred_idx.cpu().numpy().tolist())
        test_y_true.extend(labels.cpu().numpy().tolist())
        test_p_true.extend(defer_label.cpu().numpy().tolist())
        test_p_pred.extend(p_pred_idx.cpu().numpy().tolist())
        
    test_acc = accuracy_score(test_y_true, test_y_pred)
    test_qwk = cohen_kappa_score(test_y_true, test_y_pred, weights='quadratic')
    test_f1 = f1_score(test_y_true, test_y_pred, average='macro')
    test_policy_acc = accuracy_score(test_p_true, test_p_pred)
    test_policy_f1 = f1_score(test_p_true, test_p_pred, average='macro')
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Test QWK: {:.4f}".format(test_qwk))
    print("Test F1: {:.4f}".format(test_f1))
    print("Test Policy Accuracy: {:.4f}".format(test_policy_acc))
    print("Test Policy F1: {:.4f}".format(test_policy_f1))
    
    wandb.log({
        "test_acc": test_acc,
        "test_qwk": test_qwk,
        "test_f1": test_f1,
        "test_policy_acc": test_policy_acc,
        "test_policy_f1": test_policy_f1
    })
    wandb.finish()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_name', type=str, default='mnli_joint_ckpt',
                        help='Checkpoint name')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (e.g., cuda:0 or cpu)')
    args = parser.parse_args()
    train(args)
    
if __name__ == '__main__':
    main()
