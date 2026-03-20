import torch
import torch.nn as nn   
from tqdm import tqdm
from utils.util import (evaluate_annotations, compute_propagated_metrics, FocalLoss,get_cosine_schedule_with_warmup)
import os
from datetime import datetime
from model import CustomModel
import math
from test_zero import (identify_unseen_labels, print_unseen_label_analysis,
                      evaluate_unseen_labels, compute_harmonic_mean)
from torch.nn import DataParallel

def create_model_and_optimizer(config, train_domain_features, pos_weight=None, total_steps=None, adj=None):
    model = CustomModel(
        esm_dim=config['embed_dim'],
        nlp_dim=config['nlp_dim'],
        domain_size=train_domain_features.shape[1],
        hidden_dim=config.get('hidden_dim', 512),
        dropout=config.get('dropout', 0.5)
    ).cuda()
    
    if config['loss']=='focal':
        criterion = FocalLoss()
    elif  config['loss']=='bce_weight':
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif config['loss']=='bce':
        criterion = nn.BCEWithLogitsLoss()  

    if config['optimizer']=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif  config['optimizer']=='adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    if total_steps is None:
        total_steps = 1000 
    
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=config.get('num_cycles', 0.5)
    )

    print("\n" + "="*50)
    print("Model Configuration:")
    print("="*50)
    print(f"ESM Embedding Dim: {config['embed_dim']}")
    print(f"NLP Embedding Dim: {config['nlp_dim']}")
    print(f"Domain Feature Size: {train_domain_features.shape[1]}")
    print(f"Hidden Dim: {config.get('hidden_dim', 512)}")
    print(f"Dropout: {config.get('dropout', 0.3)}")
    print(f"\nTrainable parameters:")
    
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")
            trainable_params += param.numel()
    
    print(f"\nTotal trainable parameters: {trainable_params:,}")
    print(f"Estimated total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Learning rate schedule: Warmup + Cosine Annealing")
    print("="*50 + "\n")

    return model, criterion, optimizer, scheduler


def train_one_epoch_efficient(model, train_dataloader, list_embedding, criterion, optimizer, 
                              scheduler, epoch, key):
    model.train()
    loss_mean = 0
    
    list_embedding = list_embedding.cuda()  # [num_go_terms, nlp_dim]
    num_go_terms = list_embedding.shape[0]
    
    for batch_idx, batch_data in tqdm(enumerate(train_dataloader), 
                                    desc=f"Epoch {epoch+1} Training",
                                    total=len(train_dataloader)):
        optimizer.zero_grad()
        
        batch_embeddings = batch_data['embedding'].cuda()
        batch_domain_features = batch_data['domain_feature'].cuda()
        batch_labels = batch_data['labels'].cuda()
        batch_size = batch_embeddings.shape[0]
        
        esm_expanded = batch_embeddings.unsqueeze(1).expand(-1, num_go_terms, -1)
        domain_expanded = batch_domain_features.unsqueeze(1).expand(-1, num_go_terms, -1)
        
        esm_flat = esm_expanded.reshape(-1, esm_expanded.size(-1))
        domain_flat = domain_expanded.reshape(-1, domain_expanded.size(-1))
        
        outputs = model(esm_flat, domain_flat, list_embedding, batch_size)
        
        loss = criterion(outputs, batch_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        loss_mean += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            print('{}  Epoch [{}], Step [{}/{}], LR: {:.6f}, Loss: {:.4f}'.format(
                key, epoch + 1, batch_idx + 1,
                len(train_dataloader), current_lr, loss_mean / (batch_idx + 1)))
    
    avg_loss = loss_mean / len(train_dataloader)
    print(f"\nEpoch {epoch+1} Training Summary:")
    print(f"  Avg Training Loss: {avg_loss:.4f}")
    
    return avg_loss


def evaluate_model_with_unseen(model, test_dataloader, list_embedding, ia_list, key, 
                               adj_matrix, unseen_indices, seen_indices):
    model.eval()
    _labels = []
    _preds = []
    sigmoid = torch.nn.Sigmoid()
    
    list_embedding = list_embedding.cuda()
    num_go_terms = list_embedding.shape[0]
    
    print(f"\n{'='*80}")
    print(f"Evaluating {key} on test set (with unseen label analysis)...")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader, desc=f"Evaluating {key}"):
            batch_embeddings = batch_data['embedding'].cuda()
            batch_domain_features = batch_data['domain_feature'].cuda()
            batch_labels = batch_data['labels']
            batch_size = batch_embeddings.shape[0]
            
            esm_expanded = batch_embeddings.unsqueeze(1).expand(-1, num_go_terms, -1)
            domain_expanded = batch_domain_features.unsqueeze(1).expand(-1, num_go_terms, -1)
            
            esm_flat = esm_expanded.reshape(-1, esm_expanded.size(-1))
            domain_flat = domain_expanded.reshape(-1, domain_expanded.size(-1))
            
            output = model(esm_flat, domain_flat, list_embedding, batch_size)
            
            output = sigmoid(output).cpu()
            
            _labels.append(batch_labels)
            _preds.append(output)
    
    all_labels = torch.cat(_labels, dim=0)
    all_preds = torch.cat(_preds, dim=0)
    
    # Overall evaluation
    f, p, r, aupr, th = evaluate_annotations(all_labels, all_preds)
    prop_fmax, prop_precision, prop_recall, prop_aupr, prop_th, prop_preds = compute_propagated_metrics(
        all_labels, all_preds, adj_matrix
    )
    
    print(f"\n{'='*80}")
    print(f"Overall Results for {key}:")
    print(f"{'='*80}")
    print(f"  Avg Fmax:          {100 * f:.2f}%★")
    print(f"  Avg Precision:     {100 * p:.2f}%")
    print(f"  Avg Recall:        {100 * r:.2f}%")
    print(f"  AUPR:              {100 * aupr:.2f}%")
    print(f"  Threshold:         {th:.4f}")
    print(f"  Prop-Fmax:         {100 * prop_fmax:.2f}% ")
    print(f"  Prop-Precision:    {100 * prop_precision:.2f}%")
    print(f"  Prop-Recall:       {100 * prop_recall:.2f}%")
    print(f"  Prop-AUPR:         {100 * prop_aupr:.2f}%")
    print(f"  Prop-Threshold:    {prop_th:.4f}")
    
    # Unseen/Seen label analysis
    label_specific_metrics = evaluate_unseen_labels(
        all_labels, all_preds, unseen_indices, seen_indices, adj_matrix
    )
    
    # Print unseen label metrics
    if label_specific_metrics['unseen'] is not None:
        unseen = label_specific_metrics['unseen']
        print(f"\n{'='*80}")
        print(f"Unseen Labels Performance ({unseen['count']} labels, {unseen['sample_count']} samples):")
        print(f"{'='*80}")
        print(f"  Avg Fmax:          {100 * unseen['Fmax']:.2f}% ★")
        print(f"  Avg Precision:     {100 * unseen['precision']:.2f}%")
        print(f"  Avg Recall:        {100 * unseen['recall']:.2f}%")
        print(f"  AUPR:              {100 * unseen['aupr']:.2f}%")
        print(f"  Threshold:         {unseen['threshold']:.4f}")
        if 'prop_Fmax' in unseen:
            print(f"  Prop-Fmax:         {100 * unseen['prop_Fmax']:.2f}%")
            print(f"  Prop-Precision:    {100 * unseen['prop_precision']:.2f}%")
            print(f"  Prop-Recall:       {100 * unseen['prop_recall']:.2f}%")
            print(f"  Prop-AUPR:         {100 * unseen['prop_aupr']:.2f}%")
            print(f"  Prop-Threshold:    {unseen['prop_threshold']:.4f}")
    else:
        print(f"\nNo unseen labels with positive samples")
    
    # Print seen label metrics
    if label_specific_metrics['seen'] is not None:
        seen = label_specific_metrics['seen']
        print(f"\n{'='*80}")
        print(f"Seen Labels Performance ({seen['count']} labels, {seen['sample_count']} samples):")
        print(f"{'='*80}")
        print(f"  Avg Fmax:          {100 * seen['Fmax']:.2f}%★")
        print(f"  Avg Precision:     {100 * seen['precision']:.2f}%")
        print(f"  Avg Recall:        {100 * seen['recall']:.2f}%")
        print(f"  AUPR:              {100 * seen['aupr']:.2f}%")
        print(f"  Threshold:         {seen['threshold']:.4f}")
        if 'prop_Fmax' in seen:
            print(f"  Prop-Fmax:         {100 * seen['prop_Fmax']:.2f}% ")
            print(f"  Prop-Precision:    {100 * seen['prop_precision']:.2f}%")
            print(f"  Prop-Recall:       {100 * seen['prop_recall']:.2f}%")
            print(f"  Prop-AUPR:         {100 * seen['prop_aupr']:.2f}%")
            print(f"  Prop-Threshold:    {seen['prop_threshold']:.4f}")
    else:
        print(f"\nNo seen labels with positive samples")
    
    # Calculate harmonic mean
    harmonic_mean = None
    if (label_specific_metrics['unseen'] is not None and 
        label_specific_metrics['seen'] is not None and
        'aupr' in label_specific_metrics['unseen'] and
        'aupr' in label_specific_metrics['seen']):
        
        harmonic_mean = compute_harmonic_mean(
            label_specific_metrics['unseen']['aupr'],
            label_specific_metrics['seen']['aupr']
        )
        
        print(f"\n{'='*80}")
        print(f"Harmonic Mean (H):")
        print(f"  H = {100 * harmonic_mean:.2f}% ★★")
    
    print(f"{'='*80}\n")
    
    metrics = {
        'p': p,
        'r': r,
        'Fmax': f,
        'aupr': aupr,
        'threshold': th,
        'prop_Fmax': prop_fmax,
        'prop_precision': prop_precision,
        'prop_recall': prop_recall,
        'prop_aupr': prop_aupr,
        'prop_threshold': prop_th,
        'unseen': label_specific_metrics['unseen'],
        'seen': label_specific_metrics['seen'],
        'harmonic_mean': harmonic_mean
    }
    
    return metrics


def train_model_for_ontology(config, key, train_dataloader, test_dataloader, 
                            list_embedding, ia_list, ctime, 
                            metrics_output_test, train_domain_features, 
                            adj_matrix=None, pos_weight=None,
                            training_labels_binary=None, test_labels_binary=None,
                            label_list=None):
    
    max_epochs = config.get('epoch_num', 100)
    
    unseen_indices, seen_indices, train_counts, test_counts = identify_unseen_labels(
        training_labels_binary, test_labels_binary
    )
    
    print_unseen_label_analysis(key, unseen_indices, seen_indices, 
                                train_counts, test_counts, label_list)
    
    estimated_total_steps = len(train_dataloader) * max_epochs
    
    model, criterion, optimizer, scheduler = create_model_and_optimizer(
        config, train_domain_features, pos_weight, estimated_total_steps, adj_matrix
    )
    
    print(f"\n{'='*80}")
    print(f"Training {key}")
    
    for epoch in range(max_epochs):
        train_loss = train_one_epoch_efficient(
            model, train_dataloader, list_embedding, criterion, 
            optimizer, scheduler, epoch, key
        )
    
    print(f"\n{'='*80}")
    print(f"Training completed for {key}!")
    print(f"Starting final evaluation with unseen label analysis...")
    
    metrics = evaluate_model_with_unseen(
        model, test_dataloader, list_embedding, ia_list, key, 
        adj_matrix, unseen_indices, seen_indices
    )
    
    if key not in metrics_output_test:
        metrics_output_test[key] = {}
    
    for metric_name, metric_value in metrics.items():
        metrics_output_test[key][metric_name] = metric_value
    
    metrics_output_test[key]['unseen_count'] = len(unseen_indices)
    metrics_output_test[key]['seen_count'] = len(seen_indices)
    metrics_output_test[key]['total_epochs'] = max_epochs
    
    ckpt_dir = './ckpt/cafa5/MZSGO/'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{ctime}MZSGO_{key}_final.pt")
    
    torch.save(model.state_dict(), ckpt_path)

    print(f"\n{'='*80}")
    print(f"Model saved:")
    print(f"  Overall Avg Fmax: {metrics['Fmax']:.4f}")
    if metrics['harmonic_mean'] is not None:
        print(f"  Harmonic Mean: {metrics['harmonic_mean']:.4f} ★★")
    
    return model