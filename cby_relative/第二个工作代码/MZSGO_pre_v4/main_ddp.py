"""
MZSGO-DA 下游训练主入口 —— End-to-End 双Adapter + 拼接融合。

核心设计：
1. 不再提取静态embedding，直接end-to-end训练
2. 双Adapter策略：adapter_0(冻结,预训练域功能先验) + adapter_1(可训练,任务适配)
3. ESM2在forward中参与计算，但参数不更新
4. 序列特征与GO标签文本嵌入拼接后送入分类MLP
5. 对序列特征施加模态丢弃（p=0.15）

使用方式：
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 torchrun --nproc_per_node=7 main_ddp.py \
        --run_mode full --onto all --epoch_num 30 \
        --pretrain_ckpt ./ckpt/pretrain/pretrain_best.pt
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from dataset import (obo_graph, load_datasets, process_labels_for_ontology,
                     create_ontology_adjacency_matrix, compute_pos_weight)
from config import setup_environment, get_config
from go_embed import load_nlp_model, compute_nlp_embeddings_list
from util import (evaluate_annotations, compute_propagated_metrics,
                  filter_samples_with_labels, save_results, get_ontologies_to_train,
                  FocalLoss, get_cosine_schedule_with_warmup)
from test_zero import (identify_unseen_labels, print_unseen_label_analysis,
                       evaluate_unseen_labels, compute_harmonic_mean)
from model_downstream import EndToEndMZSGO


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-End MZSGO Downstream Training (v4)')
    
    parser.add_argument('--run_mode', type=str, default='full', choices=['full', 'sample'])
    parser.add_argument('--text_mode', type=str, default='all')
    parser.add_argument('--occ_num', type=int, default=0)
    parser.add_argument('--batch_size_train', type=int, default=4,
                        help='Per-GPU batch size (smaller due to ESM2 in forward)')
    parser.add_argument('--batch_size_test', type=int, default=4)
    parser.add_argument('--epoch_num', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--nlp_model_type', type=str, default='qwen_4b')
    parser.add_argument('--esm_type', type=str, default='esm2_t33_650M_UR50D')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--onto', type=str, default='all', choices=['all', 'bp', 'mf', 'cc'])
    
    # 预训练相关
    parser.add_argument('--pretrain_ckpt', type=str, default='./ckpt/pretrain/pretrain_best.pt')
    parser.add_argument('--num_adapter_layers', type=int, default=16)
    parser.add_argument('--adapter_dropout', type=float, default=0.0)
    
    # DDP
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation for memory efficiency')
    
    # 早停参数
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                        help='Minimum loss decrease to count as improvement')
    
    return parser.parse_args()


def setup_ddp():
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


class SequenceDataset(torch.utils.data.Dataset):
    """
    End-to-End 训练用的数据集。
    
    存储原始蛋白质序列（而非预提取的embedding），
    在 forward 时通过 ESM2+Adapter 编码。
    """
    def __init__(self, protein_ids, sequences, labels, alphabet, max_len=1022):
        self.protein_ids = protein_ids
        self.sequences = sequences
        self.labels = labels
        self.alphabet = alphabet
        self.max_len = max_len
        self.batch_converter = alphabet.get_batch_converter()
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        
        return {
            'protein_id': self.protein_ids[idx],
            'sequence': seq,
            'labels': self.labels[idx] if self.labels is not None else None,
        }


def sequence_collate_fn(batch, alphabet):
    """将序列转为 ESM2 token"""
    batch_converter = alphabet.get_batch_converter()
    
    protein_ids = [item['protein_id'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    _, _, batch_tokens = batch_converter(list(zip(protein_ids, sequences)))
    
    if labels[0] is not None:
        labels_tensor = torch.stack([torch.tensor(l) if not isinstance(l, torch.Tensor) else l 
                                      for l in labels])
    else:
        labels_tensor = None
    
    return {
        'tokens': batch_tokens,
        'labels': labels_tensor,
    }


def train_one_epoch(model, train_dataloader, list_embedding, criterion,
                    optimizer, scheduler, scaler, epoch, key, args, rank):
    """End-to-End 训练一个 epoch"""
    model.train()
    loss_mean = 0
    num_batches = 0
    
    list_embedding_cuda = list_embedding.cuda()
    
    if is_main_process(rank):
        pbar = tqdm(enumerate(train_dataloader),
                    desc=f"Epoch {epoch+1} Training ({key})",
                    total=len(train_dataloader))
    else:
        pbar = enumerate(train_dataloader)
    
    for batch_idx, batch_data in pbar:
        tokens = batch_data['tokens'].cuda(non_blocking=True)
        batch_labels = batch_data['labels'].cuda(non_blocking=True)
        batch_size = tokens.shape[0]
        
        is_accumulation_step = (batch_idx + 1) % args.gradient_accumulation_steps != 0
        
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # ★ End-to-End: 直接传入 tokens，模型内部经过 ESM2+Adapter
        if args.fp16:
            with torch.cuda.amp.autocast():
                outputs = model(tokens, list_embedding_cuda, batch_size)
                loss = criterion(outputs, batch_labels)
            scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
        else:
            outputs = model(tokens, list_embedding_cuda, batch_size)
            loss = criterion(outputs, batch_labels)
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
        
        if not is_accumulation_step or (batch_idx + 1) == len(train_dataloader):
            if args.fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()
        
        loss_mean += loss.item()
        num_batches += 1
        
        if is_main_process(rank) and isinstance(pbar, tqdm):
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
    
    avg_loss = loss_mean / max(num_batches, 1)
    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss], device='cuda')
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    
    return avg_loss


def evaluate_model(model, test_dataloader, list_embedding, ia_list, key,
                   adj_matrix, unseen_indices, seen_indices, rank):
    """评估模型"""
    model.eval()
    _labels = []
    _preds = []
    sigmoid = torch.nn.Sigmoid()
    
    list_embedding_cuda = list_embedding.cuda()
    
    if is_main_process(rank):
        pbar = tqdm(test_dataloader, desc=f"Evaluating {key}")
    else:
        pbar = test_dataloader
    
    with torch.no_grad():
        for batch_data in pbar:
            tokens = batch_data['tokens'].cuda(non_blocking=True)
            batch_labels = batch_data['labels']
            batch_size = tokens.shape[0]
            
            base_model = model.module if isinstance(model, DDP) else model
            output = base_model(tokens, list_embedding_cuda, batch_size)
            output = sigmoid(output).cpu()
            
            _labels.append(batch_labels)
            _preds.append(output)
    
    all_labels = torch.cat(_labels, dim=0)
    all_preds = torch.cat(_preds, dim=0)
    
    if not is_main_process(rank):
        return None
    
    f, p, r, aupr, th = evaluate_annotations(all_labels, all_preds)
    prop_fmax, prop_precision, prop_recall, prop_aupr, prop_th, _ = compute_propagated_metrics(
        all_labels, all_preds, adj_matrix
    )
    
    print(f"\n{'='*80}")
    print(f"Overall Results for {key}:")
    print(f"  Fmax: {100*f:.2f}%★  AUPR: {100*aupr:.2f}%")
    print(f"  Prop-Fmax: {100*prop_fmax:.2f}%  Prop-AUPR: {100*prop_aupr:.2f}%")
    
    label_metrics = evaluate_unseen_labels(
        all_labels, all_preds, unseen_indices, seen_indices, adj_matrix
    )
    
    if label_metrics['unseen'] is not None:
        print(f"  Unseen({label_metrics['unseen']['count']}): "
              f"Fmax={100*label_metrics['unseen']['Fmax']:.2f}% "
              f"AUPR={100*label_metrics['unseen']['aupr']:.2f}%")
    if label_metrics['seen'] is not None:
        print(f"  Seen({label_metrics['seen']['count']}): "
              f"Fmax={100*label_metrics['seen']['Fmax']:.2f}% "
              f"AUPR={100*label_metrics['seen']['aupr']:.2f}%")
    
    harmonic_mean = None
    if label_metrics['unseen'] is not None and label_metrics['seen'] is not None:
        harmonic_mean = compute_harmonic_mean(
            label_metrics['unseen']['aupr'], label_metrics['seen']['aupr']
        )
        print(f"  Harmonic Mean (H): {100*harmonic_mean:.2f}%★★")
    
    return {
        'p': p, 'r': r, 'Fmax': f, 'aupr': aupr, 'threshold': th,
        'prop_Fmax': prop_fmax, 'prop_precision': prop_precision, 'prop_recall': prop_recall,
        'prop_aupr': prop_aupr, 'prop_threshold': prop_th,
        'unseen': label_metrics['unseen'],
        'seen': label_metrics['seen'],
        'unseen_count': len(unseen_indices) if unseen_indices is not None else 0,
        'seen_count': len(seen_indices) if seen_indices is not None else 0,
        'harmonic_mean': harmonic_mean
    }


def main():
    args = parse_args()
    seed = setup_environment()
    local_rank, rank, world_size = setup_ddp()
    
    config = get_config(
        run_mode=args.run_mode,
        text_mode=args.text_mode,
        occ_num=args.occ_num,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        nlp_model_type=args.nlp_model_type,
        epoch_num=args.epoch_num,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        esm_type=args.esm_type,
        loss=args.loss,
    )
    
    ctime = datetime.now().strftime("%Y%m%d%H%M%S")
    
    if is_main_process(rank):
        print(f"\n{'='*80}")
        print(f"MZSGO-DA End-to-End Downstream Training")
        print(f"{'='*80}")
        print(f"Key features:")
        print(f"  - End-to-End training (no static embeddings)")
        print(f"  - Dual Adapter: adapter_0(frozen,pretrained) + adapter_1(trainable)")
        print(f"  - Concatenation fusion (seq || label → MLP)")
        print(f"  - Pretrain checkpoint: {args.pretrain_ckpt}")
        print(f"  - Gradient accumulation: {args.gradient_accumulation_steps}")
    
    ontologies_to_train = get_ontologies_to_train(args.onto)
    
    # 加载 NLP 模型和 GO 本体
    nlp_model, nlp_tokenizer = load_nlp_model(config)
    
    label_space = {
        'biological_process': [],
        'molecular_function': [],
        'cellular_component': []
    }
    enc = preprocessing.LabelEncoder()
    onto, ia_dict = obo_graph(config['obo_path'], config['ia_path'])
    
    train_id, training_sequences, training_labels, test_id, test_sequences, test_labels = \
        load_datasets(config, onto, label_space)
    
    metrics_output_test = {}
    
    for key in ontologies_to_train:
        if is_main_process(rank):
            print(f"\n{'='*80}")
            print(f"Processing ontology: {key}")
        
        label_list, training_labels_binary, test_labels_binary, enc, ia_list, onto_parent, label_num = \
            process_labels_for_ontology(config, key, label_space, training_labels, test_labels, onto, enc, ia_dict)
        
        # 过滤无标签样本（不需要 embedding 参数了）
        filtered_data = filter_samples_with_labels(
            training_labels_binary, test_labels_binary,
            training_sequences, test_sequences,
            None, None,  # ★ 不再需要预提取的 embedding
            None, None,
            train_id, test_id
        )
        
        if filtered_data is None:
            continue
        
        adj_matrix = create_ontology_adjacency_matrix(onto_parent, label_num, key, config)
        pos_weight = compute_pos_weight(filtered_data['train']['labels']).cuda()
        
        list_nlp = compute_nlp_embeddings_list(
            config, nlp_model, nlp_tokenizer, key, label_list, onto).cuda()
        
        unseen_indices, seen_indices, train_counts, test_counts = identify_unseen_labels(
            training_labels_binary, test_labels_binary
        )
        if is_main_process(rank):
            print_unseen_label_analysis(key, unseen_indices, seen_indices,
                                        train_counts, test_counts, label_list)
        
        # ★ 创建每个本体独立的 EndToEndMZSGO 模型
        model = EndToEndMZSGO(
            esm_type=args.esm_type,
            nlp_dim=config['nlp_dim'],
            hidden_dim=config.get('hidden_dim', 512),
            dropout=config.get('dropout', 0.3),
            num_adapter_layers=args.num_adapter_layers,
            adapter_dropout=args.adapter_dropout,
            pretrain_adapter_ckpt=args.pretrain_ckpt,
        ).cuda()
        
        if is_main_process(rank):
            model.print_trainable_params()
        
        # DDP 包装
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
        
        # 创建 DataLoader（使用原始序列）
        from functools import partial
        base_model = model.module
        collate_fn = partial(sequence_collate_fn, alphabet=base_model.alphabet)
        
        train_ids = filtered_data['train'].get('ids', 
            [f"train_{i}" for i in range(len(filtered_data['train']['sequences']))])
        test_ids = filtered_data['test'].get('ids',
            [f"test_{i}" for i in range(len(filtered_data['test']['sequences']))])
        
        train_dataset = SequenceDataset(
            train_ids,
            filtered_data['train']['sequences'],
            filtered_data['train']['labels'],
            base_model.alphabet,
        )
        test_dataset = SequenceDataset(
            test_ids,
            filtered_data['test']['sequences'],
            filtered_data['test']['labels'],
            base_model.alphabet,
        )
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size_train,
            sampler=train_sampler, num_workers=4, pin_memory=True,
            drop_last=True, collate_fn=collate_fn
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size_test,
            shuffle=False, num_workers=4, pin_memory=True,
            collate_fn=collate_fn
        )
        
        # 优化器（只优化可训练参数：adapter_1 + 分类头）
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
        
        total_steps = (len(train_dataloader) // args.gradient_accumulation_steps) * args.epoch_num
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
        # 损失函数
        if args.loss == 'focal':
            criterion = FocalLoss()
        elif args.loss == 'bce_weight':
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
        
        # 早停追踪
        best_train_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # 训练
        for epoch in range(args.epoch_num):
            train_sampler.set_epoch(epoch)
            train_loss = train_one_epoch(
                model, train_dataloader, list_nlp, criterion,
                optimizer, scheduler, scaler, epoch, key, args, rank
            )
            if is_main_process(rank):
                print(f"  Epoch {epoch+1}/{args.epoch_num} - Loss: {train_loss:.4f}", end="")
                
                if train_loss < best_train_loss - args.min_delta:
                    best_train_loss = train_loss
                    patience_counter = 0
                    # 保存最佳模型状态
                    base_model_save = model.module if isinstance(model, DDP) else model
                    best_model_state = {k: v.cpu().clone() for k, v in base_model_save.state_dict().items()}
                    print(f" ★ Best (patience reset)")
                else:
                    patience_counter += 1
                    print(f" (patience: {patience_counter}/{args.patience})")
            
            # 广播早停信号到所有进程
            stop_signal = torch.tensor([0], device='cuda')
            if is_main_process(rank):
                if patience_counter >= args.patience:
                    stop_signal[0] = 1
            dist.broadcast(stop_signal, src=0)
            
            if stop_signal.item() == 1:
                if is_main_process(rank):
                    print(f"\n  Early stopping at epoch {epoch+1}! Best loss: {best_train_loss:.4f}")
                break
        
        # 恢复最佳模型用于评估
        if best_model_state is not None and is_main_process(rank):
            base_model_load = model.module if isinstance(model, DDP) else model
            base_model_load.load_state_dict(best_model_state)
            if is_main_process(rank):
                print(f"  Restored best model (loss={best_train_loss:.4f}) for evaluation")
        
        # 评估
        metrics = evaluate_model(
            model, test_dataloader, list_nlp, ia_list, key,
            adj_matrix, unseen_indices, seen_indices, rank
        )
        
        if is_main_process(rank) and metrics is not None:
            metrics_output_test[key] = metrics
            
            ckpt_dir = './ckpt/cafa5/MZSGO_DA/'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"{ctime}MZSGO_DA_{key}_final.pt")
            base_model = model.module if isinstance(model, DDP) else model
            torch.save(base_model.state_dict(), ckpt_path)
            print(f"Model saved to {ckpt_path}")
        
        dist.barrier()
        
        # 释放模型显存
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
    
    if is_main_process(rank):
        save_results(config, metrics_output_test, seed, ctime)
        print(f'\nEnd: {datetime.now().strftime("%Y%m%d%H%M%S")}')
    
    cleanup_ddp()


if __name__ == "__main__":
    main()
