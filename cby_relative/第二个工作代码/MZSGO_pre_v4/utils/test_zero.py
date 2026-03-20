from datetime import datetime
import torch
import os
import numpy as np
from util import evaluate_annotations, compute_propagated_metrics

def identify_unseen_labels(train_labels, test_labels):
    
    if isinstance(train_labels, list):
        train_labels = np.array(train_labels, dtype=np.float32)
    elif isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.cpu().numpy()
    
    if isinstance(test_labels, list):
        test_labels = np.array(test_labels, dtype=np.float32)
    elif isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.cpu().numpy()
    
    train_label_counts = train_labels.sum(axis=0)
    
    test_label_counts = test_labels.sum(axis=0)
    
    unseen_mask = (train_label_counts == 0) & (test_label_counts > 0)
    unseen_label_indices = np.where(unseen_mask)[0].tolist()
    
    seen_mask = (train_label_counts > 0) & (test_label_counts > 0)
    seen_label_indices = np.where(seen_mask)[0].tolist()
    
    train_label_counts = torch.from_numpy(train_label_counts)
    test_label_counts = torch.from_numpy(test_label_counts)
    
    return unseen_label_indices, seen_label_indices, train_label_counts, test_label_counts


def evaluate_unseen_labels(all_labels, all_preds, unseen_indices, seen_indices, adj_matrix=None):
    metrics = {}
    
    if len(unseen_indices) > 0:
        unseen_labels = all_labels[:, unseen_indices]
        unseen_preds = all_preds[:, unseen_indices]
        
        if unseen_labels.sum().item() > 0:
            f, p, r, aupr, th = evaluate_annotations(unseen_labels, unseen_preds,"unseen")
            
            metrics['unseen'] = {
                'count': len(unseen_indices),
                'Fmax': f,
                'precision': p,
                'recall': r,
                'aupr': aupr,
                'threshold': th,
                'sample_count': int(unseen_labels.sum().item())
            }
            
            if adj_matrix is not None:
                try:
                    if adj_matrix.is_sparse:
                        adj_matrix_dense = adj_matrix.to_dense()
                    else:
                        adj_matrix_dense = adj_matrix
                    
                    unseen_adj = adj_matrix_dense[unseen_indices][:, unseen_indices]
                    
                    if adj_matrix.is_sparse:
                        unseen_adj = unseen_adj.to_sparse()
                    
                    prop_Fmax, prop_precision, prop_recall, prop_aupr, prop_th, _ = compute_propagated_metrics(
                        unseen_labels, unseen_preds, unseen_adj
                    )
                    metrics['unseen']['prop_Fmax'] = prop_Fmax
                    metrics['unseen']['prop_precision'] = prop_precision
                    metrics['unseen']['prop_recall'] = prop_recall
                    metrics['unseen']['prop_aupr'] = prop_aupr
                    metrics['unseen']['prop_threshold'] = prop_th
                except Exception as e:
                    print(f"Warning: Could not compute propagated metrics for unseen labels: {e}")
        else:
            print("Warning: No positive samples for unseen labels in test set")
            metrics['unseen'] = None
    else:
        metrics['unseen'] = None
    
    if len(seen_indices) > 0:
        seen_labels = all_labels[:, seen_indices]
        seen_preds = all_preds[:, seen_indices]
        
        if seen_labels.sum().item() > 0:
            f, p, r, aupr, th = evaluate_annotations(seen_labels, seen_preds,"seen")
            
            metrics['seen'] = {
                'count': len(seen_indices),
                'Fmax': f,
                'precision': p,
                'recall': r,
                'aupr': aupr,
                'threshold': th,
                'sample_count': int(seen_labels.sum().item())
            }
            
            if adj_matrix is not None:
                try:
                    if adj_matrix.is_sparse:
                        adj_matrix_dense = adj_matrix.to_dense()
                    else:
                        adj_matrix_dense = adj_matrix
                    
                    seen_adj = adj_matrix_dense[seen_indices][:, seen_indices]
                    
                    if adj_matrix.is_sparse:
                        seen_adj = seen_adj.to_sparse()
                    
                    prop_Fmax, prop_precision, prop_recall, prop_aupr, prop_th, _ = compute_propagated_metrics(
                        seen_labels, seen_preds, seen_adj
                    )
                    metrics['seen']['prop_Fmax'] = prop_Fmax
                    metrics['seen']['prop_precision'] = prop_precision
                    metrics['seen']['prop_recall'] = prop_recall
                    metrics['seen']['prop_aupr'] = prop_aupr
                    metrics['seen']['prop_threshold'] = prop_th
                except Exception as e:
                    print(f"Warning: Could not compute propagated metrics for seen labels: {e}")
        else:
            print("Warning: No positive samples for seen labels in test set")
            metrics['seen'] = None
    else:
        metrics['seen'] = None
    
    return metrics

def compute_harmonic_mean(unseen_aupr, seen_aupr):
    if unseen_aupr is None or seen_aupr is None:
        return None
    
    if unseen_aupr == 0 and seen_aupr == 0:
        return 0.0
    
    if unseen_aupr == 0 or seen_aupr == 0:
        return 0.0
    
    harmonic_mean = (2 * unseen_aupr * seen_aupr) / (unseen_aupr + seen_aupr)
    return harmonic_mean

def print_unseen_label_analysis(key, unseen_indices, seen_indices, train_counts, test_counts, label_list):
    print(f"\n{'='*80}")
    print(f"Label Analysis for {key}")
    print(f"  Total labels:        {len(label_list)}")
    print(f"  Unseen labels:       {len(unseen_indices)} (appear only in test set)")
    print(f"  Seen labels:         {len(seen_indices)} (appear in both train and test)")
    
    train_only = ((train_counts > 0) & (test_counts == 0)).sum().item()
    print(f"  Train-only labels:   {train_only}")
    
    if len(unseen_indices) > 0:
        print(f"\n  Unseen Label Details (top 10):")
        print(f"  {'Index':<8} {'GO Term':<15} {'Test Count':<12}")
        print(f"  {'-'*40}")
        
        unseen_test_counts = [(idx, test_counts[idx].item()) for idx in unseen_indices]
        unseen_test_counts.sort(key=lambda x: x[1], reverse=True)
        
        for idx, count in unseen_test_counts[:10]:  
            go_term = label_list[idx] if idx < len(label_list) else "N/A"
            print(f"  {idx:<8} {go_term:<15} {int(count):<12}")
        
        if len(unseen_indices) > 10:
            print(f"  ... and {len(unseen_indices) - 10} more")
        
        total_unseen_samples = sum(count for _, count in unseen_test_counts)
        print(f"\n  Total unseen label annotations in test set: {int(total_unseen_samples)}")
    
    print(f"{'='*80}\n")

def save_test_results(config, all_results, output_dir='./test_results'):
    os.makedirs(output_dir, exist_ok=True)
    ctime = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = os.path.join(output_dir, f"{config['model']}_{config['run_mode']}_{ctime}.txt")
    
    with open(output_file, 'w') as f:
        f.write(f"OVERALL TEST RESULTS\n")
        f.write(f"{'='*80}\n")
        
        for key, result in all_results.items():
            metrics = result['metrics']
            checkpoint_path = result['checkpoint_path']
            
            f.write(f"\n{key}:\n")
            f.write(f"  Model path:        {checkpoint_path}\n")
            f.write(f"  Avg Precision:     {metrics['p']:.4f}\n")
            f.write(f"  Avg Recall:        {metrics['r']:.4f}\n")
            f.write(f"  Avg Fmax:          {metrics['Fmax']:.4f}\n ★")
            f.write(f"  AUPR:              {metrics['aupr']:.4f}\n")
            f.write(f"  Threshold:         {metrics['threshold']:.4f}\n")
            f.write(f"  Prop-Fmax:         {metrics['prop_Fmax']:.4f}\n")
            f.write(f"  Prop-Precision:    {metrics['prop_precision']:.4f}\n")
            f.write(f"  Prop-Recall:       {metrics['prop_recall']:.4f}\n")
            f.write(f"  Prop-AUPR:         {metrics['prop_aupr']:.4f}\n")
            f.write(f"  Prop-Threshold:    {metrics['prop_threshold']:.4f}\n")
            if metrics['harmonic_mean'] is not None:
                f.write(f"  Harmonic Mean (H): {metrics['harmonic_mean']:.4f} ★★\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"UNSEEN LABELS PERFORMANCE (Zero-shot)\n")
        f.write(f"{'='*80}\n")
        
        for key, result in all_results.items():
            metrics = result['metrics']
            unseen = metrics.get('unseen')
            
            f.write(f"\n{key}:\n")
            if unseen is not None:
                f.write(f"  Label count:       {unseen['count']}\n")
                f.write(f"  Sample count:      {unseen['sample_count']}\n")
                f.write(f"  Avg Precision:     {unseen['precision']:.4f}\n")
                f.write(f"  Avg Recall:        {unseen['recall']:.4f}\n")
                f.write(f"  Avg Fmax:          {unseen['Fmax']:.4f} ★\n")
                f.write(f"  AUPR:              {unseen['aupr']:.4f}\n")
                f.write(f"  Threshold:         {unseen['threshold']:.4f}\n")
                if 'prop_Fmax' in unseen:
                    f.write(f"  Prop-Fmax:         {unseen['prop_Fmax']:.4f}\n")
                    f.write(f"  Prop-Precision:    {unseen['prop_precision']:.4f}\n")
                    f.write(f"  Prop-Recall:       {unseen['prop_recall']:.4f}\n")
                    f.write(f"  Prop-AUPR:         {unseen['prop_aupr']:.4f}\n")
                    f.write(f"  Prop-Threshold:    {unseen['prop_threshold']:.4f}\n")
            else:
                f.write(f"  No unseen labels with positive samples\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"SEEN LABELS PERFORMANCE\n")
        f.write(f"{'='*80}\n")
        
        for key, result in all_results.items():
            metrics = result['metrics']
            seen = metrics.get('seen')
            
            f.write(f"\n{key}:\n")
            if seen is not None:
                f.write(f"  Label count:       {seen['count']}\n")
                f.write(f"  Sample count:      {seen['sample_count']}\n")
                f.write(f"  Avg Precision:     {seen['precision']:.4f}\n")
                f.write(f"  Avg Recall:        {seen['recall']:.4f}\n")
                f.write(f"  Avg Fmax:          {seen['Fmax']:.4f} ★\n")
                f.write(f"  AUPR:              {seen['aupr']:.4f}\n")
                f.write(f"  Threshold:         {seen['threshold']:.4f}\n")
                if 'prop_Fmax' in seen:
                    f.write(f"  Prop-Fmax:         {seen['prop_Fmax']:.4f}\n")
                    f.write(f"  Prop-Precision:    {seen['prop_precision']:.4f}\n")
                    f.write(f"  Prop-Recall:       {seen['prop_recall']:.4f}\n")
                    f.write(f"  Prop-AUPR:         {seen['prop_aupr']:.4f}\n")
                    f.write(f"  Prop-Threshold:    {seen['prop_threshold']:.4f}\n")
            else:
                f.write(f"  No seen labels with positive samples\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"HARMONIC MEAN (H)\n")
        f.write(f"{'='*80}\n")
        
        for key, result in all_results.items():
            metrics = result['metrics']
            f.write(f"\n{key}:\n")
            if metrics['harmonic_mean'] is not None:
                f.write(f"  H = {metrics['harmonic_mean']:.4f} ★★\n")
            else:
                f.write(f"  Cannot compute harmonic mean (missing unseen or seen labels)\n")
    
    print(f"\n{'='*80}")
    print(f"Test results saved to: {output_file}")
