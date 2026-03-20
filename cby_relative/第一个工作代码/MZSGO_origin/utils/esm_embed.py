from math import e
import os
import pickle
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import esm
import sys
from scipy.sparse import csr_matrix, save_npz, load_npz
from collections import defaultdict
from collections import Counter
_esm_model = None
_esm_tokenizer = None
_esm_num_layers = None

def load_esm_model(config):
    """Load ESM model"""
    global _esm_model, _esm_tokenizer, _esm_num_layers
    
    if _esm_model is None:
        print("Loading ESM model...")
        
        esm_type = config.get('esm_type', 'esm2_t33_650M_UR50D')
        
        if esm_type == 'esm2_t33_650M_UR50D':
            _esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            _esm_num_layers = 33
        elif esm_type == 'esm2_t36_3B_UR50D':
            _esm_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            _esm_num_layers = 36
        elif esm_type == 'esm2_t48_15B_UR50D':
            _esm_model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
            _esm_num_layers = 48
        else:
            _esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            _esm_num_layers = 33
        
        _esm_tokenizer = alphabet
        _esm_model = _esm_model.cuda()
        _esm_model.eval()
        
        print(f"ESM model loaded: {esm_type}")
    
    return _esm_model, _esm_tokenizer, _esm_num_layers

def precompute_esm_embeddings(sequences, cache_file, pooling='mean'):
    cache_dir = os.path.dirname(cache_file)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_file):
        print(f"Loading cached ESM embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Computing ESM embeddings for {len(sequences)} sequences...")
    batch_converter = tokenizer.get_batch_converter()
    embeddings = []
    
    for i, seq in enumerate(tqdm(sequences, desc="Computing ESM embeddings")):
        batch_labels, batch_strs, batch_tokens = batch_converter([("x", seq)])
        with torch.no_grad():
            batch_tokens = batch_tokens.cuda()
            results = model(batch_tokens, repr_layers=[num_layers])
            token_representations = results["representations"][num_layers]
            
            plm_embed = token_representations[0, 1:1 + len(seq), :].cpu()
            
            if pooling == 'mean':
                pooled_embed = plm_embed.mean(dim=0)  # [embed_dim]
            elif pooling == 'max':
                pooled_embed, _ = plm_embed.max(dim=0)  # [embed_dim]
            elif pooling == 'cls':
                pooled_embed = token_representations[0, 0, :].cpu()  # [embed_dim]
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
            
            embeddings.append(pooled_embed)
    
    print(f"Saving ESM embeddings to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

def compute_esm_embeddings(config, training_sequences, test_sequences):
    print("Computing ESM embeddings...")
    
    if config['run_mode'] == "sample":
        train_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/train_esm_embeddings_sample.pkl")
        test_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/test_esm_embeddings_sample.pkl")
    elif config['run_mode'] == "full":
        train_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/train_esm_embeddings_mean.pkl")
        test_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/test_esm_embeddings_mean.pkl")
    elif config['run_mode'] == "zero":
        train_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/train_esm_embeddings_mean.pkl")
        test_esm_cache = os.path.join(config['cache_dir'], f"esm/{config['esm_type']}/test_esm_embeddings_zero.pkl")
    train_esm_embeddings = precompute_esm_embeddings(training_sequences, train_esm_cache, pooling='mean')
    test_esm_embeddings = precompute_esm_embeddings(test_sequences, test_esm_cache, pooling='mean')
    
    return train_esm_embeddings, test_esm_embeddings

def compute_esm_embeddings_single(sequence, config, pooling='mean'):
    """Compute ESM embeddings for a single sequence"""
    model, tokenizer, num_layers = load_esm_model(config)
    batch_converter = tokenizer.get_batch_converter()
    
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
    
    with torch.no_grad():
        batch_tokens = batch_tokens.cuda()
        results = model(batch_tokens, repr_layers=[num_layers])
        token_representations = results["representations"][num_layers]
        
        plm_embed = token_representations[0, 1:1 + len(sequence), :].cpu()
        
        if pooling == 'mean':
            pooled_embed = plm_embed.mean(dim=0)
        elif pooling == 'max':
            pooled_embed, _ = plm_embed.max(dim=0)
        elif pooling == 'cls':
            pooled_embed = token_representations[0, 0, :].cpu()
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
    
    return pooled_embed