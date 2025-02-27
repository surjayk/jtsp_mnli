# main_mnli.py

import argparse
import wandb
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from transformers import get_linear_schedule_with_warmup

from DataModules_mnli import create_dataloaders
from SFRNModel_mnli import SFRNModel, DeferralClassifier
from Constants_mnli import hyperparameters, config_dictionary

def evaluate(model, policy, loader, device, criterion_cl, criterion_p):

    model.eval()
    policy.eval()

    all_labels = []
    all_preds = []
    policy_labels = []
    policy_preds = []

    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            embedding_ids = batch["embedding_ids"].to(device)
            embedding_mask = batch["embedding_mask"].to(device)
            labels = batch["label"].to(device)

            logits, hidden = model(input_ids, attention_mask)
            loss_cl = criterion_cl(logits, labels)

            preds = torch.argmax(logits, dim=1)
            defer_label = (preds != labels).long()

            defer_logits = policy(embedding_ids, hidden, attention_mask=embedding_mask)
            loss_defer = criterion_p(defer_logits, defer_label)

            loss = loss_cl + loss_defer
            total_loss += loss.item()

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            policy_labels.extend(defer_label.cpu().tolist())
            policy_preds.extend(torch.argmax(defer_logits, dim=1).cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    policy_acc = accuracy_score(policy_labels, policy_preds)
    policy_f1 = f1_score(policy_labels, policy_preds, average="macro")

    return {
        "loss": total_loss / len(loader),
        "acc": acc,
        "f1": f1,
        "qwk": qwk,
        "policy_acc": policy_acc,
        "policy_f1": policy_f1
    }


def train(args):
    wandb.init(project="mnli_experiment", config=config_dictionary)
    random.seed(hyperparameters['random_seed'])
    np.random.seed(hyperparameters['random_seed'])
    torch.manual_seed(hyperparameters['random_seed'])

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    hyperparameters["p_lr"] = args.p_lr
    hyperparameters["lr"] = args.clf_lr
    hyperparameters["policy_hidden"] = args.policy_hidden
    hyperparameters["alpha"] = args.alpha
    hyperparameters["beta"] = args.beta
    hyperparameters["train_id"] = args.run_name


    train_loader, val_loader, test_loader = create_dataloaders()


    model = SFRNModel(num_labels=hyperparameters['num_labels']).to(device)
    policy = DeferralClassifier(
        input_dim=hyperparameters['mlp_hidden'],  # model hidden size
        hidden_dim=hyperparameters['policy_hidden'],
        output_dim=2
    ).to(device)

    if hyperparameters["CL_CHECKPOINT_PATH"]:
        model.load_state_dict(torch.load(hyperparameters["CL_CHECKPOINT_PATH"], map_location=device))
        print(f"Loaded classifier checkpoint from {hyperparameters['CL_CHECKPOINT_PATH']}")

    optimizer = AdamW(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
    optimizer_p = AdamW(policy.parameters(), lr=hyperparameters['p_lr'], weight_decay=hyperparameters['weight_decay'])

    criterion_cl = nn.CrossEntropyLoss()
    criterion_p = nn.CrossEntropyLoss()

    num_training_steps = len(train_loader) * hyperparameters['epochs']
    warmup_steps = int(hyperparameters['WARMUP_STEPS'] * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    scheduler_p = get_linear_schedule_with_warmup(optimizer_p, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    pre_step = hyperparameters.get("pre_step", 1)     # classifier-only warm-up
    mid_step = hyperparameters.get("mid_step", 2)     # policy-only warm-up
    total_epochs = hyperparameters['epochs']

    alpha = hyperparameters["alpha"]  # weight for classifier loss
    beta  = hyperparameters["beta"]   # weight for policy loss

    grad_accum_steps = hyperparameters.get("GRADIENT_ACCUMULATION_STEPS", 1)
    best_acc, best_f1 = 0.0, 0.0
    best_path = None

    print(f"Phases => pre_step={pre_step}, mid_step={mid_step}, total_epochs={total_epochs}")

    for epoch in range(total_epochs):
        # Phase logic
        if epoch < pre_step:
            # classifier only
            model.train()
            policy.eval()
            phase_desc = "Classifier-Only"
        elif epoch < mid_step:
            # policy only
            model.eval()
            policy.train()
            phase_desc = "Policy-Only"
        else:
            # joint
            model.train()
            policy.train()
            phase_desc = "Joint"

        print(f"\n=== EPOCH {epoch+1}/{total_epochs} ({phase_desc} phase) ===")

        train_loss, train_cl_loss, train_policy_loss = 0.0, 0.0, 0.0
        y_true, y_pred = [], []
        p_true, p_pred = [], []

        # Training loop
        optimizer.zero_grad()
        optimizer_p.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            embedding_ids = batch["embedding_ids"].to(device)
            embedding_mask = batch["embedding_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass: Classifier
            logits, hidden = model(input_ids, attention_mask)
            loss_cl = criterion_cl(logits, labels)

            # Determine deferral label
            preds = torch.argmax(logits, dim=1)
            defer_label = (preds != labels).long()

            # Forward pass: Deferral policy
            defer_logits = policy(embedding_ids, hidden, attention_mask=embedding_mask)
            loss_defer = criterion_p(defer_logits, defer_label)

            # Which losses to use depending on phase
            if epoch < pre_step:
                # classifier only
                loss = loss_cl
            elif epoch < mid_step:
                # policy only
                loss = loss_defer
            else:
                # joint
                loss = alpha * loss_cl + beta * loss_defer

            # Gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()

            train_loss += loss.item()
            train_cl_loss += loss_cl.item() / grad_accum_steps
            train_policy_loss += loss_defer.item() / grad_accum_steps

            # Collect metrics
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            p_true.extend(defer_label.cpu().tolist())
            p_pred.extend(torch.argmax(defer_logits, dim=1).cpu().tolist())

            if (step + 1) % grad_accum_steps == 0:
                # Clip grad if needed
                nn.utils.clip_grad_norm_(model.parameters(), hyperparameters['max_norm'])
                nn.utils.clip_grad_norm_(policy.parameters(), hyperparameters['max_norm'])

                if epoch < pre_step:
                    # Update classifier only
                    optimizer.step()
                    optimizer.zero_grad()
                    # policy stays in eval mode, but we can zero its grad
                    optimizer_p.zero_grad()
                    scheduler.step()
                elif epoch < mid_step:
                    # Update policy only
                    optimizer_p.step()
                    optimizer_p.zero_grad()
                    # classifier stays in eval mode, but we can zero its grad
                    optimizer.zero_grad()
                    scheduler_p.step()
                else:
                    # Joint update
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizer_p.step()
                    optimizer_p.zero_grad()
                    scheduler.step()
                    scheduler_p.step()

        train_acc = accuracy_score(y_true, y_pred)
        train_f1  = f1_score(y_true, y_pred, average="macro")
        train_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

        policy_acc = accuracy_score(p_true, p_pred)
        policy_f1  = f1_score(p_true, p_pred, average="macro")

        print(f"[Train] loss={train_loss:.4f} cl_loss={train_cl_loss:.4f} policy_loss={train_policy_loss:.4f} ")
        print(f"        acc={train_acc:.3f}, f1={train_f1:.3f}, qwk={train_qwk:.3f}")
        print(f"        policy_acc={policy_acc:.3f}, policy_f1={policy_f1:.3f}")

        # Validation
        val_metrics = evaluate(model, policy, val_loader, device, criterion_cl, criterion_p)
        print(f"[Val]   loss={val_metrics['loss']:.4f}  acc={val_metrics['acc']:.3f}, f1={val_metrics['f1']:.3f}, qwk={val_metrics['qwk']:.3f}")
        print(f"        policy_acc={val_metrics['policy_acc']:.3f}, policy_f1={val_metrics['policy_f1']:.3f}")

        # Track best model by validation accuracy (or f1)
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            best_f1  = val_metrics["f1"]
            best_path = f"checkpoint/{hyperparameters['train_id']}_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"[*] Best model updated => {best_path}")

        # Log to W&B
        wandb.log({
            "epoch": epoch+1,
            "phase": phase_desc,
            "train_loss": train_loss,
            "train_cl_loss": train_cl_loss,
            "train_policy_loss": train_policy_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "train_qwk": train_qwk,
            "train_policy_acc": policy_acc,
            "train_policy_f1": policy_f1,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "val_qwk": val_metrics["qwk"],
            "val_policy_acc": val_metrics["policy_acc"],
            "val_policy_f1": val_metrics["policy_f1"]
        })

    # Training complete
    print("\nTraining complete!")
    print(f"Best model path: {best_path}, val_acc={best_acc:.4f}, val_f1={best_f1:.4f}")

    # Final Test (using best checkpoint if available)
    if best_path is not None:
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, policy, test_loader, device, criterion_cl, criterion_p)
    print(f"[Test]  acc={test_metrics['acc']:.3f}, f1={test_metrics['f1']:.3f}, qwk={test_metrics['qwk']:.3f}")
    print(f"        policy_acc={test_metrics['policy_acc']:.3f}, policy_f1={test_metrics['policy_f1']:.3f}")

    wandb.log({
        "final_test_acc": test_metrics["acc"],
        "final_test_f1": test_metrics["f1"],
        "final_test_qwk": test_metrics["qwk"],
        "final_test_policy_acc": test_metrics["policy_acc"],
        "final_test_policy_f1": test_metrics["policy_f1"]
    })
    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="Device: 'cpu' or 'cuda:0'")
    parser.add_argument("--run_name", type=str, default="jtsp_mnli", help="Unique run name for checkpoint/logging")
    parser.add_argument("--p_lr", type=float, default=1e-5, help="Policy learning rate")
    parser.add_argument("--clf_lr", type=float, default=2e-5, help="Classifier learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for classifier loss")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for policy loss")
    parser.add_argument("--policy_hidden", type=int, default=256, help="Deferral classifier hidden dimension")
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
