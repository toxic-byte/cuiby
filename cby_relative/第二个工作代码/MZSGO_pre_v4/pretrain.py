"""
预训练主脚本（v4版本）：使用DDP进行多GPU分布式训练。

v4改进：
- 使用内嵌Adapter的ESM2（正常梯度反传）
- 额外保存adapter_state_dict，用于下游任务加载

使用方式：
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7 torchrun --nproc_per_node=7 pretrain.py \
        --batch_size 8 --epochs 50 --lr 1e-4
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from pretrain_model import DomainAwarePretrainModel
from pretrain_dataset import build_pretrain_dataset, create_pretrain_dataloaders
from config import setup_environment, get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Domain-Aware Contrastive Pretraining (v4)')
    
    # 数据参数
    parser.add_argument('--run_mode', type=str, default='full', choices=['full', 'sample'])
    parser.add_argument('--nlp_model_type', type=str, default='qwen_4b')
    parser.add_argument('--esm_type', type=str, default='esm2_t33_650M_UR50D')
    
    # 模型参数
    parser.add_argument('--num_adapter_layers', type=int, default=16)
    parser.add_argument('--bottleneck_dim', type=int, default=None)
    parser.add_argument('--projection_hidden_dim', type=int, default=512)
    parser.add_argument('--projection_output_dim', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--adapter_dropout', type=float, default=0.0,
                        help='Adapter dropout (S-PLM uses 0)')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fp16', action='store_true', default=True)
    
    # 早停参数
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                        help='Minimum loss decrease to count as improvement')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./ckpt/pretrain')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='./logs/pretrain')
    parser.add_argument('--log_every', type=int, default=50)
    
    # DDP参数
    parser.add_argument('--local_rank', type=int, default=-1)
    
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


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, 
                                      num_cycles=0.5):
    import math
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, epoch, 
                    args, writer, rank, global_step):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    if is_main_process(rank):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    else:
        pbar = dataloader
    
    for batch_idx, batch in enumerate(pbar):
        tokens = batch['tokens'].cuda(non_blocking=True)
        domain_embeddings = batch['domain_embeddings'].cuda(non_blocking=True)
        
        is_accumulation_step = (batch_idx + 1) % args.gradient_accumulation_steps != 0
        
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        if args.fp16:
            with torch.cuda.amp.autocast():
                loss, seq_proj, domain_proj = model(tokens, domain_embeddings)
            scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
        else:
            loss, seq_proj, domain_proj = model(tokens, domain_embeddings)
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
        
        if not is_accumulation_step or (batch_idx + 1) == len(dataloader):
            if args.fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        if is_main_process(rank):
            if isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
            
            if global_step % args.log_every == 0 and writer is not None:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
                
                with torch.no_grad():
                    sim_matrix = torch.matmul(seq_proj, domain_proj.T)
                    diag_sim = sim_matrix.diag().mean().item()
                    off_diag_mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, 
                                                device=sim_matrix.device)
                    off_diag_sim = sim_matrix[off_diag_mask].mean().item()
                    
                    writer.add_scalar('train/pos_similarity', diag_sim, global_step)
                    writer.add_scalar('train/neg_similarity', off_diag_sim, global_step)
                    writer.add_scalar('train/sim_gap', diag_sim - off_diag_sim, global_step)
    
    avg_loss = total_loss / max(num_batches, 1)
    loss_tensor = torch.tensor([avg_loss], device='cuda')
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    avg_loss = loss_tensor.item()
    
    return avg_loss, global_step


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, 
                    avg_loss, save_path, args):
    model_to_save = model.module if isinstance(model, DDP) else model
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'avg_loss': avg_loss,
        'args': vars(args),
        # ★ v4: 额外保存 adapter 的 state_dict，方便下游加载
        'adapter_state_dict': model_to_save.get_adapter_state_dict(),
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    args = parse_args()
    seed = setup_environment()
    local_rank, rank, world_size = setup_ddp()
    
    if is_main_process(rank):
        print(f"\n{'='*80}")
        print(f"Domain-Aware Contrastive Pretraining (v4 - Internal Adapter)")
        print(f"{'='*80}")
        print(f"World size: {world_size}")
        print(f"Key v4 improvements:")
        print(f"  - Adapter embedded INSIDE Transformer layers (attn后 + FFN后)")
        print(f"  - Clean residual: module(x) + x (no LayerNorm wrapping)")
        print(f"  - Normal gradient backprop (no FrozenLayerForward/detach)")
        print(f"  - Adapter dropout: {args.adapter_dropout}")
        print(f"Arguments: {vars(args)}")
    
    config = get_config(
        run_mode=args.run_mode,
        nlp_model_type=args.nlp_model_type,
        esm_type=args.esm_type,
    )
    
    dataset = build_pretrain_dataset(config)
    domain_dim = dataset.domain_features.shape[1]
    
    model = DomainAwarePretrainModel(
        esm_type=args.esm_type,
        domain_dim=domain_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_output_dim=args.projection_output_dim,
        num_adapter_layers=args.num_adapter_layers,
        bottleneck_dim=args.bottleneck_dim,
        adapter_dropout=args.adapter_dropout,
        temperature=args.temperature,
    ).cuda()
    
    if is_main_process(rank):
        model.print_trainable_params()
    
    # DDP包装
    # ★ v4: 因为 ESM2 参数冻结但仍在计算图中，
    # 而 Adapter 参数嵌入在被替换的层内部，DDP 需要 find_unused_parameters
    # 来处理 lm_head、contact_head 等不参与反传的模块
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)
    
    dataloader, sampler = create_pretrain_dataloaders(
        dataset=dataset,
        alphabet=model.module.seq_encoder.alphabet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=True,
        world_size=world_size,
        rank=rank
    )
    
    if is_main_process(rank):
        print(f"\nDataLoader created:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"  Steps per epoch: {len(dataloader)}")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    writer = None
    if is_main_process(rank):
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    global_step = 0
    best_loss = float('inf')
    patience_counter = 0
    
    if is_main_process(rank):
        print(f"\nStarting training for {args.epochs} epochs (early stopping patience={args.patience})...")
        print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}\n")
    
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        avg_loss, global_step = train_one_epoch(
            model, dataloader, optimizer, scheduler, scaler,
            epoch, args, writer, rank, global_step
        )
        
        if is_main_process(rank):
            print(f"\nEpoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f}")
            
            if writer is not None:
                writer.add_scalar('epoch/avg_loss', avg_loss, epoch)
            
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                save_path = os.path.join(args.save_dir, f'pretrain_epoch{epoch+1}.pt')
                save_checkpoint(model, optimizer, scheduler, scaler, 
                              epoch, global_step, avg_loss, save_path, args)
            
            if avg_loss < best_loss - args.min_delta:
                best_loss = avg_loss
                patience_counter = 0
                save_path = os.path.join(args.save_dir, 'pretrain_best.pt')
                save_checkpoint(model, optimizer, scheduler, scaler,
                              epoch, global_step, avg_loss, save_path, args)
                print(f"  ★ New best loss: {best_loss:.4f} (patience reset)")
            else:
                patience_counter += 1
                print(f"  No improvement (patience: {patience_counter}/{args.patience})")
        
        # 广播早停信号到所有进程
        stop_signal = torch.tensor([0], device='cuda')
        if is_main_process(rank):
            if patience_counter >= args.patience:
                stop_signal[0] = 1
        dist.broadcast(stop_signal, src=0)
        
        if stop_signal.item() == 1:
            if is_main_process(rank):
                print(f"\n{'='*80}")
                print(f"Early stopping triggered at epoch {epoch+1}!")
                print(f"Best loss: {best_loss:.4f}")
                print(f"{'='*80}")
            break
        
        dist.barrier()
    
    if is_main_process(rank):
        print(f"\n{'='*80}")
        print(f"Training completed! Best loss: {best_loss:.4f}")
        if patience_counter < args.patience:
            print(f"Completed all {args.epochs} epochs (no early stopping)")
        else:
            print(f"Early stopped at epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")
        if writer is not None:
            writer.close()
    
    cleanup_ddp()


if __name__ == '__main__':
    main()
