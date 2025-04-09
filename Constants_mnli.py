
hyperparameters = dict(
    train_id="1217_joint_bert_ro_mnli_sp",  # updated train id for MNLI joint training
    model_name="bert-base-uncased",
    defer_model_name="roberta-base",
    emb_name="curie-001",
    num_labels=3,
    max_length=128,
    random_seed=23,
    data_split=0.2,      
    lr=1e-5,
    p_lr=1e-7,
    epochs=20,
    pre_step=5,
    mid_step=5,
    weight_decay=0.01,
    GRADIENT_ACCUMULATION_STEPS=32,
    max_norm=1,
    WARMUP_STEPS=0.2,
    hidden_dropout_prob=0.2,
    hidden_dim=768,
    p_hidden_dim=768,
    mlp_hidden=256,
    gpt_dim=1536,
    policy_hidden=256,
    alpha=0.5,
    beta=0.1,
    gamma=1,
    SUBSET_SIZE=392702,   # full MNLI train split size
    TRAIN_SIZE=379702,    # portion used for training after holding out internal test set
    TEST_SIZE=13000,      # internal test subset from training data
    VAL_SIZE=9800,        # carved out from the official validation set
    batch_size=16
)

config_dictionary = dict(params=hyperparameters)
