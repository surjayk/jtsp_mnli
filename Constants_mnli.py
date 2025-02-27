# Constants_mnli.py
hyperparameters = dict(
    train_id="joint_mnli_run",
    model_name="bert-base-uncased",
    defer_model_name="bert-base-uncased",
    num_labels=3,
    max_length=128,
    random_seed=42,
    epochs=60,
    pre_step=23,
    mid_step=30,
    lr=2e-5,          # default classifier LR
    p_lr=1e-5,        # default policy LR
    weight_decay=0.01,
    GRADIENT_ACCUMULATION_STEPS=1,
    max_norm=1,
    WARMUP_STEPS=0.2,
    batch_size=16,
    hidden_dropout_prob=0.1,
    hidden_dim=768,
    mlp_hidden=256,
    policy_hidden=256,
    alpha=0.5,
    beta=0.5,
    gamma=1,
    CL_CHECKPOINT_PATH="/data/smk6961/jtsp/SFRN_mnli/checkpoint/checkpoint_sfrn_mnli_pretrain_best.pth",
    SUBSET_SIZE=392702,
    TRAIN_SIZE=379702,
    TEST_SIZE=13000,
    VAL_SIZE=9800
)

config_dictionary = dict(params=hyperparameters)
