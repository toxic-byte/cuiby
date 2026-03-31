"""
预训练数据集（MZSGO-DA）：
为双通道对比学习构建(蛋白质序列, domain embedding, GO func embedding)三元组数据。

关键设计：
- 保留所有蛋白质样本（包括缺乏域注释的）
- 缺乏域注释的样本仅参与"序列-功能"通道，不参与"序列-域"通道
- 通过 has_domain 标记区分
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
    双通道对比学习预训练数据集。
    
    每个样本包含：
    - sequence: 蛋白质氨基酸序列字符串
    - domain_embedding: 该蛋白质所有domain embedding的聚合向量（可能为零向量）
    - func_embedding: 该蛋白质所有GO功能标签embedding的聚合向量
    - has_domain: bool，是否有域注释
    - protein_id: 蛋白质ID
    """
    def __init__(self, protein_ids, sequences, domain_features, func_features, max_len=1022):
        """
        Args:
            protein_ids: list of str
            sequences: list of str (氨基酸序列)
            domain_features: torch.Tensor [N, domain_dim] 或 np.ndarray
            func_features: torch.Tensor [N, func_dim] 或 np.ndarray
            max_len: ESM2的最大序列长度（去掉BOS/EOS后）
        """
        assert len(protein_ids) == len(sequences) == len(domain_features) == len(func_features)
        
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
        
        # 确保func_features是tensor
        if isinstance(func_features, np.ndarray):
            self.func_features = torch.FloatTensor(func_features)
        elif isinstance(func_features, torch.Tensor):
            self.func_features = func_features.float()
        else:
            self.func_features = torch.FloatTensor(np.array(func_features))
        
        # ★ 标记哪些样本有域注释（非零向量）
        self.has_domain = self.domain_features.abs().sum(dim=-1) > 0  # [N] bool
        
        # ★ 过滤掉func embedding也为零的样本（没有任何GO注释的蛋白质无法参与任何通道）
        func_valid = self.func_features.abs().sum(dim=-1) > 0
        valid_indices = torch.where(func_valid)[0].tolist()
        
        original_count = len(protein_ids)
        self.protein_ids = [protein_ids[i] for i in valid_indices]
        self.sequences = [sequences[i] for i in valid_indices]
        self.domain_features = self.domain_features[valid_indices]
        self.func_features = self.func_features[valid_indices]
        self.has_domain = self.has_domain[valid_indices]
        
        domain_count = self.has_domain.sum().item()
        no_domain_count = len(self.protein_ids) - domain_count
        
        print(f"PretrainDataset: {original_count} -> {len(self.protein_ids)} proteins "
              f"(filtered {original_count - len(self.protein_ids)} without GO annotations)")
        print(f"  With domain annotations: {domain_count}")
        print(f"  Without domain annotations (func-only): {no_domain_count}")
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        
        return {
            'protein_id': self.protein_ids[idx],
            'sequence': seq,
            'domain_embedding': self.domain_features[idx],
            'func_embedding': self.func_features[idx],
            'has_domain': self.has_domain[idx],
        }


def pretrain_collate_fn(batch, alphabet):
    """
    自定义collate函数，将蛋白质序列转为ESM2 token。
    """
    batch_converter = alphabet.get_batch_converter()
    
    protein_ids = [item['protein_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    domain_embeddings = torch.stack([item['domain_embedding'] for item in batch])
    func_embeddings = torch.stack([item['func_embedding'] for item in batch])
    has_domain = torch.stack([item['has_domain'] for item in batch])
    
    # 使用ESM2的batch_converter处理序列
    batch_labels, batch_strs, batch_tokens = batch_converter(
        list(zip(protein_ids, sequences))
    )
    
    return {
        'protein_ids': protein_ids,
        'tokens': batch_tokens,  # [B, L] (包含BOS/EOS)
        'domain_embeddings': domain_embeddings,  # [B, domain_dim]
        'func_embeddings': func_embeddings,  # [B, func_dim]
        'has_domain': has_domain,  # [B] bool
    }


def build_pretrain_dataset(config):
    """
    构建预训练数据集。
    
    使用完整Swiss-Prot数据库（排除下游测试集）中的蛋白质序列、
    domain信息和GO功能信息，组装成双通道对比学习所需的数据三元组。
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
    
    # ★ 使用独立的预训练数据文件（完整Swiss-Prot排除测试集）
    pretrain_path = config.get('pretrain_path', config['train_path'])
    print(f"Pretrain data source: {pretrain_path}")
    
    from dataset import preprocess_dataset
    train_id, training_sequences, training_labels = preprocess_dataset(
        pretrain_path, config['MAXLEN'], onto, label_space
    )
    
    # 加载domain features
    train_domain_features, _ = load_text_pretrained_domain_features(
        train_id, train_id, config  # test_id用train_id占位
    )
    
    # 只取train部分的domain_features
    train_domain_features = train_domain_features[:len(train_id)]
    
    # ★ 加载GO功能嵌入：为每个蛋白质的所有GO注释聚合功能嵌入
    train_func_features = build_protein_func_embeddings(
        train_id, training_labels, onto, config
    )
    
    dataset = PretrainDataset(
        protein_ids=train_id,
        sequences=training_sequences,
        domain_features=train_domain_features,
        func_features=train_func_features,
        max_len=min(config.get('MAXLEN', 1022), 1022)  # ESM2最大1022，避免OOM
    )
    
    print(f"Pretrain dataset size: {len(dataset)}")
    print(f"Domain embedding dim: {dataset.domain_features.shape[1]}")
    print(f"Func embedding dim: {dataset.func_features.shape[1]}")
    
    return dataset


def build_protein_func_embeddings(protein_ids, training_labels, onto, config):
    """
    为每个蛋白质构建GO功能嵌入。
    
    对于拥有多个GO注释的蛋白质，将各GO标签的文本嵌入经平均池化聚合为蛋白质级功能表示。
    使用Qwen3-Embedding预计算。
    
    Args:
        protein_ids: list of protein IDs
        training_labels: dict of {namespace: list of list of GO terms}
        onto: list of ontology objects
        config: config dict
        
    Returns:
        func_features: torch.Tensor [N, nlp_dim]
    """
    import re
    
    nlp_dim = config.get('nlp_dim', 2560)
    cache_dir = config.get('cache_dir', './data/embeddings_cache')
    nlp_model_type = config.get('nlp_model_type', 'qwen_4b')
    run_mode = config.get('run_mode', 'full')
    
    cache_path = os.path.join(cache_dir, f"pretrain_go/protein_go_embeddings_{nlp_model_type}_{run_mode}.pkl")
    
    if os.path.exists(cache_path):
        print(f"Loading cached protein func embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        if len(cached_data) == len(protein_ids):
            print(f"Cache loaded. Shape: [{len(cached_data)}, {nlp_dim}]")
            return torch.FloatTensor(np.array(cached_data))
        else:
            print(f"Cache size mismatch ({len(cached_data)} vs {len(protein_ids)}), regenerating...")
    
    print("Building protein-level GO func embeddings...")
    
    # 收集所有蛋白质的GO注释（合并三个本体）
    protein_go_terms = []
    for i in range(len(protein_ids)):
        all_terms = set()
        for ns_key in ['biological_process', 'molecular_function', 'cellular_component']:
            if ns_key in training_labels and i < len(training_labels[ns_key]):
                terms = training_labels[ns_key][i]
                for term in terms:
                    all_terms.add(term)
        protein_go_terms.append(list(all_terms))
    
    # 收集所有唯一的GO term，获取其文本描述
    all_unique_terms = set()
    for terms in protein_go_terms:
        all_unique_terms.update(terms)
    all_unique_terms = sorted(list(all_unique_terms))
    
    print(f"Total unique GO terms across all proteins: {len(all_unique_terms)}")
    
    # 获取每个GO term的文本描述
    term_descriptions = {}
    for term in all_unique_terms:
        term_with_prefix = 'GO:' + term if not term.startswith('GO:') else term
        description = None
        for ont in onto:
            if term_with_prefix in ont.terms_dict:
                term_info = ont.terms_dict[term_with_prefix]
                name_part = term_info.get('name', '')
                def_part = ''
                tag_context = term_info.get('def', '')
                if tag_context:
                    tag_contents = re.findall(r'"(.*?)"', tag_context)
                    if tag_contents:
                        def_part = tag_contents[0]
                if name_part and def_part:
                    description = f"{name_part}: {def_part}"
                elif name_part:
                    description = name_part
                elif def_part:
                    description = def_part
                break
        if description:
            term_descriptions[term] = description
    
    print(f"GO terms with descriptions: {len(term_descriptions)}/{len(all_unique_terms)}")
    
    # 使用NLP模型编码所有GO term
    go_term_embeddings_cache = os.path.join(cache_dir, f"pretrain_go/all_go_term_embeddings_{nlp_model_type}_{run_mode}.pkl")
    
    if os.path.exists(go_term_embeddings_cache):
        print(f"Loading cached GO term embeddings from {go_term_embeddings_cache}")
        with open(go_term_embeddings_cache, 'rb') as f:
            go_term_embeddings = pickle.load(f)
    else:
        from go_embed import load_nlp_model
        nlp_model, nlp_tokenizer = load_nlp_model(config)
        
        go_term_embeddings = {}
        descriptions_list = []
        terms_list = []
        for term in all_unique_terms:
            if term in term_descriptions:
                terms_list.append(term)
                descriptions_list.append(term_descriptions[term])
        
        print(f"Encoding {len(descriptions_list)} GO term descriptions...")
        batch_size = 64
        for i in tqdm(range(0, len(descriptions_list), batch_size), desc="Encoding GO terms"):
            batch_texts = descriptions_list[i:i + batch_size]
            batch_terms = terms_list[i:i + batch_size]
            
            encoded = nlp_tokenizer(
                batch_texts,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                input_ids = encoded['input_ids'].cuda()
                attention_mask = encoded['attention_mask'].cuda()
                outputs = nlp_model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
            for j, term in enumerate(batch_terms):
                go_term_embeddings[term] = embeddings[j]
        
        # 保存GO term embeddings缓存
        os.makedirs(os.path.dirname(go_term_embeddings_cache), exist_ok=True)
        with open(go_term_embeddings_cache, 'wb') as f:
            pickle.dump(go_term_embeddings, f)
        print(f"Saved GO term embeddings cache to {go_term_embeddings_cache}")
    
    # 为每个蛋白质聚合GO功能嵌入（平均池化）
    print("Aggregating per-protein GO func embeddings...")
    func_features = []
    no_func_count = 0
    
    for i, terms in enumerate(protein_go_terms):
        valid_embeddings = []
        for term in terms:
            if term in go_term_embeddings:
                valid_embeddings.append(go_term_embeddings[term])
        
        if len(valid_embeddings) > 0:
            aggregated = np.mean(valid_embeddings, axis=0)
        else:
            aggregated = np.zeros(nlp_dim, dtype=np.float32)
            no_func_count += 1
        
        func_features.append(aggregated)
    
    func_features = np.array(func_features, dtype=np.float32)
    
    print(f"Protein func embeddings shape: {func_features.shape}")
    print(f"Proteins without GO func embeddings: {no_func_count}/{len(protein_ids)}")
    
    # 保存缓存
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(func_features.tolist(), f)
    print(f"Saved protein func embeddings cache to {cache_path}")
    
    return torch.FloatTensor(func_features)


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
