"""
预训练数据集：为对比学习构建(蛋白质序列, domain embedding)配对数据。
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pickle
import numpy as np
import os
import sys
from tqdm import tqdm

# 确保utils目录在搜索路径中
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))


class PretrainDataset(Dataset):
    """
    对比学习预训练数据集。
    
    每个样本包含：
    - sequence: 蛋白质氨基酸序列字符串
    - domain_embedding: 该蛋白质所有domain embedding的聚合向量
    - protein_id: 蛋白质ID
    """
    def __init__(self, protein_ids, sequences, domain_features, max_len=1022):
        """
        Args:
            protein_ids: list of str
            sequences: list of str (氨基酸序列)
            domain_features: torch.Tensor [N, domain_dim] 或 np.ndarray
            max_len: ESM2的最大序列长度（去掉BOS/EOS后）
        """
        assert len(protein_ids) == len(sequences) == len(domain_features)
        
        self.protein_ids = protein_ids
        self.sequences = sequences
        self.max_len = max_len
        
        # 确保domain_features是tensor
        if isinstance(domain_features, np.ndarray):
            self.domain_features = torch.FloatTensor(domain_features)
        elif isinstance(domain_features, torch.Tensor):
            self.domain_features = domain_features.float()
        else:
            self.domain_features = torch.FloatTensor(np.array(domain_features))
        
        # 过滤掉domain embedding全零的样本（没有domain注释的蛋白质）
        valid_mask = self.domain_features.abs().sum(dim=-1) > 0
        valid_indices = torch.where(valid_mask)[0].tolist()
        
        original_count = len(protein_ids)
        self.protein_ids = [protein_ids[i] for i in valid_indices]
        self.sequences = [sequences[i] for i in valid_indices]
        self.domain_features = self.domain_features[valid_indices]
        
        print(f"PretrainDataset: {original_count} -> {len(self.protein_ids)} proteins "
              f"(filtered {original_count - len(self.protein_ids)} without domain annotations)")
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        
        return {
            'protein_id': self.protein_ids[idx],
            'sequence': seq,
            'domain_embedding': self.domain_features[idx]
        }


def pretrain_collate_fn(batch, alphabet):
    """
    自定义collate函数，将蛋白质序列转为ESM2 token。
    """
    batch_converter = alphabet.get_batch_converter()
    
    protein_ids = [item['protein_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    domain_embeddings = torch.stack([item['domain_embedding'] for item in batch])
    
    # 使用ESM2的batch_converter处理序列
    batch_labels, batch_strs, batch_tokens = batch_converter(
        list(zip(protein_ids, sequences))
    )
    
    return {
        'protein_ids': protein_ids,
        'tokens': batch_tokens,  # [B, L] (包含BOS/EOS)
        'domain_embeddings': domain_embeddings  # [B, domain_dim]
    }


def build_pretrain_dataset(config):
    """
    构建预训练数据集。
    
    从MZSGO的数据文件中读取蛋白质序列和domain信息，
    组装成对比学习所需的数据对。
    """
    from dataset import preprocess_dataset, obo_graph
    from domain_embed import load_text_pretrained_domain_features
    
    print("Building pretrain dataset...")
    
    # 读取OBO图
    onto, ia_dict = obo_graph(config['obo_path'], config['ia_path'])
    
    label_space = {
        'biological_process': [],
        'molecular_function': [],
        'cellular_component': []
    }
    
    # 读取训练数据（预训练只用训练集）
    from dataset import preprocess_dataset
    train_id, training_sequences, training_labels = preprocess_dataset(
        config['train_path'], config['MAXLEN'], onto, label_space
    )
    
    # 加载domain features
    train_domain_features, _ = load_text_pretrained_domain_features(
        train_id, train_id, config  # test_id用train_id占位
    )
    
    # 只取train部分的domain_features
    train_domain_features = train_domain_features[:len(train_id)]
    
    dataset = PretrainDataset(
        protein_ids=train_id,
        sequences=training_sequences,
        domain_features=train_domain_features,
        max_len=min(config.get('MAXLEN', 1022), 1022)  # ESM2最大1022，避免OOM
    )
    
    print(f"Pretrain dataset size: {len(dataset)}")
    print(f"Domain embedding dim: {dataset.domain_features.shape[1]}")
    
    return dataset


def create_pretrain_dataloaders(dataset, alphabet, batch_size, num_workers=4, 
                                 distributed=True, world_size=1, rank=0):
    """
    创建预训练的DataLoader，支持DDP。
    """
    from functools import partial
    
    collate_fn = partial(pretrain_collate_fn, alphabet=alphabet)
    
    if distributed and world_size > 1:
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # 对比学习需要drop_last确保batch大小一致
        collate_fn=collate_fn
    )
    
    return dataloader, sampler
