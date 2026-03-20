import os
import torch
import random
import numpy as np

def setup_environment():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["WANDB_DISABLED"] = "true"
    
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed

def get_config(run_mode="full", text_mode="all", nlp_model_type="qwen_4b", occ_num=0,
batch_size_train=128, batch_size_test=128, learning_rate=5e-4, epoch_num=100, patience=10,
hidden_dim=512, model="MZSGO", dropout=0.3, esm_type="esm2_t33_650M_UR50D",
embed_dim=1280, nlp_dim=2560, loss='bce',optimizer="adam"):
    config = {
        'run_mode': run_mode,
        'text_mode': text_mode,
        'nlp_model_type': nlp_model_type,
        'occ_num': occ_num,
        'nlp_dim': nlp_dim,
        'embed_dim': embed_dim,
        'MAXLEN': 2048,
        'cache_dir': './data/embeddings_cache',
        'output_path': 'eval/',
        'obo_path': './data/go_2023_01_01.obo',
        'batch_size_train': batch_size_train,
        'batch_size_test': batch_size_test,
        'learning_rate': learning_rate,
        'epoch_num': epoch_num,
        'patience': patience,
        'dropout': dropout,
        'hidden_dim': hidden_dim,
        'domain_file_path': './data/swissprot_domains.txt',
        'esm_type': esm_type,
        'model': model,
        'loss': loss,
        'entry_path': './data/entry.list',
        'ia_path': './data/IA.txt',
        'optimizer':optimizer
    }
    
    if run_mode == "sample":
        config['train_path'] = "./data/sequence/train_sample.txt"
        config['test_path'] = "./data/sequence/test_sample.txt"
    elif run_mode == "full":
        config['train_path'] = "./data/sequence/cafa5_train_in_swissprot.txt"
        config['test_path'] = "./data/sequence/cafa5_test_in_swissprot.txt"
    elif run_mode == "zero":
        config['train_path'] = "./data/sequence/cafa5_train_in_swissprot.txt"
        config['test_path'] = "./data/sequence/zero_shot_below30.txt"
        config['obo_path'] = "./data/go_2025_10_10.obo"

    if nlp_model_type == "qwen_06b":
        config['nlp_path'] = '/d/cuiby/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418'
        config['nlp_name'] = "Qwen/Qwen3-Embedding-0.6B"
        config['nlp_dim'] = 1024
        config['domain_text_path'] = './data/embeddings_cache/domain/domain_embeddings_qwen_06b.pkl'
    elif nlp_model_type == "qwen_4b":
        config['nlp_path'] = '/d/cuiby/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b'
        config['nlp_name'] = "Qwen/Qwen3-Embedding-4B"
        config['nlp_dim'] = 2560
        config['domain_text_path'] = './data/embeddings_cache/domain/domain_embeddings_qwen_4b.pkl'
    elif nlp_model_type == "biogpt":
        config['nlp_path'] = '/d/cuiby/.cache/huggingface/hub/models--microsoft--biogpt/snapshots/eb0d815e95434dc9e3b78f464e52b899bee7d923'
        config['nlp_name'] = "microsoft/biogpt"
        config['nlp_dim'] = 1024
        config['domain_text_path'] = './data/embeddings_cache/domain/domain_embeddings_biogpt.pkl'
    
    os.makedirs(config['cache_dir'], exist_ok=True)
    return config

