import numpy as np
import torch
import torch.nn.functional as F
import os 
import math
ONTO_NAME_MAP = {
    'bp': 'biological_process',
    'mf': 'molecular_function',
    'cc': 'cellular_component'
}

def get_ontologies_to_train(onto_arg):
    if onto_arg == 'all':
        return ['biological_process', 'molecular_function', 'cellular_component']
    else:
        return [ONTO_NAME_MAP[onto_arg]]


def filter_samples_with_labels(training_labels_binary, test_labels_binary, 
                               training_sequences, test_sequences,
                               train_esm_embeddings, test_esm_embeddings,
                               train_domain_features, test_domain_features,
                               train_id, test_id):
    training_labels_binary = np.array(training_labels_binary)
    test_labels_binary = np.array(test_labels_binary)
    
    train_has_label = training_labels_binary.sum(axis=1) > 0
    train_indices = np.where(train_has_label)[0]
    
    test_has_label = test_labels_binary.sum(axis=1) > 0
    test_indices = np.where(test_has_label)[0]
    
    print(f"\n  Training samples: {len(training_sequences)} -> {len(train_indices)} (with labels)")
    print(f"  Test samples: {len(test_sequences)} -> {len(test_indices)} (with labels)")
    
    if len(train_indices) == 0:
        print(f"  WARNING: No training samples with labels for this ontology!")
        return None
    
    filtered_train_sequences = [training_sequences[i] for i in train_indices]
    filtered_train_labels = training_labels_binary[train_indices]
    filtered_train_id = [train_id[i] for i in train_indices]
    
    filtered_train_esm = None
    if train_esm_embeddings is not None:
        if isinstance(train_esm_embeddings, list):
            filtered_train_esm = [train_esm_embeddings[i] for i in train_indices]
        else:
            filtered_train_esm = train_esm_embeddings[train_indices]
    
    filtered_train_domain = None
    if train_domain_features is not None:
        if isinstance(train_domain_features, list):
            filtered_train_domain = [train_domain_features[i] for i in train_indices]
        else:
            filtered_train_domain = train_domain_features[train_indices]
    
    filtered_test_sequences = [test_sequences[i] for i in test_indices]
    filtered_test_labels = test_labels_binary[test_indices]
    filtered_test_id = [test_id[i] for i in test_indices]
    
    filtered_test_esm = None
    if test_esm_embeddings is not None:
        if isinstance(test_esm_embeddings, list):
            filtered_test_esm = [test_esm_embeddings[i] for i in test_indices]
        else:
            filtered_test_esm = test_esm_embeddings[test_indices]
    
    filtered_test_domain = None
    if test_domain_features is not None:
        if isinstance(test_domain_features, list):
            filtered_test_domain = [test_domain_features[i] for i in test_indices]
        else:
            filtered_test_domain = test_domain_features[test_indices]
    
    return {
        'train': {
            'sequences': filtered_train_sequences,
            'labels': filtered_train_labels,
            'ids': filtered_train_id,
            'esm_embeddings': filtered_train_esm,
            'domain_features': filtered_train_domain
        },
        'test': {
            'sequences': filtered_test_sequences,
            'labels': filtered_test_labels,
            'ids': filtered_test_id,
            'esm_embeddings': filtered_test_esm,
            'domain_features': filtered_test_domain
        }
    }

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()
            
def evaluate_annotations(gold, hypo, ontology_name='test'):
    temp_gold = []
    temp_hypo = []
    
    for i in range(len(gold)):
        g = gold[i].cpu().numpy() if hasattr(gold[i], 'cpu') else gold[i]
        h = hypo[i].cpu().numpy() if hasattr(hypo[i], 'cpu') else hypo[i]
        
        if np.sum(g) > 0:
            temp_gold.append(g)
            temp_hypo.append(h)
            
    print(f"{ontology_name}: {len(temp_gold)} proteins with annotations")
    
    if len(temp_gold) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    y_true = np.array(temp_gold)
    y_pred = np.array(temp_hypo)
    
    thresholds = np.linspace(0, 1, 101)
    avg_prec_list = []
    avg_rec_list = []
    f_list = []
    
    N = y_true.shape[0]
    
    for t in thresholds:
        y_pred_binary = (y_pred >= t).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred_binary == 1), axis=1)
        fp = np.sum((y_true == 0) & (y_pred_binary == 1), axis=1)
        fn = np.sum((y_true == 1) & (y_pred_binary == 0), axis=1)
        
        pred_count = tp + fp
        
        has_pred_mask = pred_count > 0
        n_with_pred = np.sum(has_pred_mask)
        
        if n_with_pred > 0:
            p_vals = tp[has_pred_mask] / pred_count[has_pred_mask]
            avg_prec = np.sum(p_vals) / n_with_pred
        else:
            avg_prec = 0.0
            
        real_count = tp + fn
        r_vals = tp / real_count
        avg_rec = np.sum(r_vals) / N 
        
        avg_prec_list.append(avg_prec)
        avg_rec_list.append(avg_rec)
        
        if (avg_prec + avg_rec) > 0:
            f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec)
        else:
            f1 = 0.0
        f_list.append(f1)
        
    fmax = max(f_list)
    max_idx = np.argmax(f_list)
    best_p = avg_prec_list[max_idx]
    best_r = avg_rec_list[max_idx]
    best_t = thresholds[max_idx]
    
    prec_array = np.array(avg_prec_list)
    rec_array = np.array(avg_rec_list)
    
    sorted_indices = np.argsort(rec_array)
    rec_sorted = rec_array[sorted_indices]
    prec_sorted = prec_array[sorted_indices]
    
    aupr = np.trapz(prec_sorted, rec_sorted)
    
    return fmax, best_p, best_r, aupr, best_t

def propagate_predictions(preds, adj_matrix):
   
    device = preds.device
    n_samples, n_labels = preds.shape
    
    if isinstance(adj_matrix, torch.Tensor):
        if adj_matrix.is_sparse:
            adj_matrix = adj_matrix.to_dense()
        adj_matrix = adj_matrix.to(device)
    else:
        adj_matrix = torch.tensor(adj_matrix, device=device)

    propagated_preds = preds.clone()
    
    for label_idx in range(n_labels):
        parents = torch.where(adj_matrix[label_idx, :] > 0)[0]
        
        if len(parents) > 0:
            child_scores = propagated_preds[:, label_idx:label_idx+1]  
            parent_scores = propagated_preds[:, parents] 
            propagated_preds[:, parents] = torch.max(parent_scores, child_scores)
    
    return propagated_preds


def compute_propagated_metrics(labels, preds, adj_matrix):
    prop_preds = propagate_predictions(preds, adj_matrix)
    
    prop_fmax, prop_precision, prop_recall, prop_aupr,prop_th = evaluate_annotations(labels, prop_preds,"prop")
    
    return prop_fmax, prop_precision,prop_recall,prop_aupr,prop_th, prop_preds

def save_results(config, metrics_output_test, seed, ctime):
    os.makedirs(config['output_path'], exist_ok=True)
    output_file = os.path.join(config['output_path'], f"{config['model']}_{config.get('text_mode', 'default')}_{ctime}.txt")
    
    with open(output_file, 'w') as file_prec:
        file_prec.write(f"FINAL EVALUATION RESULTS (with Unseen Label Analysis)\n")
        file_prec.write(f"{'='*80}\n")

        for key in metrics_output_test.keys():
            metrics = metrics_output_test[key]
            
            file_prec.write(f"\n{'='*30} {key} {'='*30}\n")
            
            file_prec.write(f"\nOverall Metrics:\n")
            file_prec.write(f"  Avg Fmax:          {metrics['Fmax']:.4f}★\n")
            file_prec.write(f"  Avg Precision:     {metrics['p']:.4f}\n")
            file_prec.write(f"  Avg Recall:        {metrics['r']:.4f}\n")
            file_prec.write(f"  AUPR:              {metrics['aupr']:.4f}\n")
            file_prec.write(f"  Threshold:         {metrics['threshold']:.4f}\n")
            file_prec.write(f"  Prop-Fmax:         {metrics['prop_Fmax']:.4f} \n")
            file_prec.write(f"  Prop-Precision:    {metrics['prop_precision']:.4f}\n")
            file_prec.write(f"  Prop-Recall:       {metrics['prop_recall']:.4f}\n")
            file_prec.write(f"  Prop-AUPR:         {metrics['prop_aupr']:.4f}\n")
            file_prec.write(f"  Prop-Threshold:    {metrics['prop_threshold']:.4f}\n")
            
            if metrics['unseen'] is not None:
                unseen = metrics['unseen']
                file_prec.write(f"\nUnseen Labels ({metrics['unseen_count']} labels, {unseen['sample_count']} samples):\n")
                file_prec.write(f"  Avg Fmax:          {unseen['Fmax']:.4f}★\n ")
                file_prec.write(f"  Avg Precision:     {unseen['precision']:.4f}\n")
                file_prec.write(f"  Avg Recall:        {unseen['recall']:.4f}\n")
                file_prec.write(f"  AUPR:              {unseen['aupr']:.4f}\n")
                file_prec.write(f"  Threshold:         {unseen['threshold']:.4f}\n")
                if 'prop_Fmax' in unseen:
                    file_prec.write(f"  Prop-Fmax:         {unseen['prop_Fmax']:.4f} \n")
                    file_prec.write(f"  Prop-Precision:    {unseen['prop_precision']:.4f}\n")
                    file_prec.write(f"  Prop-Recall:       {unseen['prop_recall']:.4f}\n")
                    file_prec.write(f"  Prop-AUPR:         {unseen['prop_aupr']:.4f}\n")
                    file_prec.write(f"  Prop-Threshold:    {unseen['prop_threshold']:.4f}\n")
            
            if metrics['seen'] is not None:
                seen = metrics['seen']
                file_prec.write(f"\nSeen Labels ({metrics['seen_count']} labels, {seen['sample_count']} samples):\n")
                file_prec.write(f"  Avg Fmax:          {seen['Fmax']:.4f} ★\n")
                file_prec.write(f"  Avg Precision:     {seen['precision']:.4f}\n")
                file_prec.write(f"  Avg Recall:        {seen['recall']:.4f}\n")
                file_prec.write(f"  AUPR:              {seen['aupr']:.4f}\n")
                file_prec.write(f"  Threshold:         {seen['threshold']:.4f}\n")
                if 'prop_Fmax' in seen:
                    file_prec.write(f"  Prop-Fmax:         {seen['prop_Fmax']:.4f}\n")
                    file_prec.write(f"  Prop-Precision:    {seen['prop_precision']:.4f}\n")
                    file_prec.write(f"  Prop-Recall:       {seen['prop_recall']:.4f}\n")
                    file_prec.write(f"  Prop-AUPR:         {seen['prop_aupr']:.4f}\n")
                    file_prec.write(f"  Prop-Threshold:    {seen['prop_threshold']:.4f}\n")
            
            if metrics['harmonic_mean'] is not None:
                file_prec.write(f"\nHarmonic Mean:     {metrics['harmonic_mean']:.4f} ★★\n")
    
    print(f"Results saved to: {output_file}")
    print(f"\n{'='*80}")

    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    for key in metrics_output_test.keys():
        metrics = metrics_output_test[key]
        print(f"\n{key}:")
        print(f"  Overall:")
        print(f"    Avg Fmax:      {metrics['Fmax']:.4f} ★")
        print(f"    Prop-Fmax:     {metrics['prop_Fmax']:.4f}")
        print(f"    AUPR:          {metrics['aupr']:.4f}")
        print(f"    Prop-AUPR:     {metrics['prop_aupr']:.4f}")
        
        if metrics['unseen'] is not None:
            print(f"  Unseen ({metrics['unseen_count']} labels):")
            print(f"    Fmax:      {metrics['unseen']['Fmax']:.4f} ★")
            print(f"    AUPR:      {metrics['unseen']['aupr']:.4f}")
        
        if metrics['seen'] is not None:
            print(f"  Seen ({metrics['seen_count']} labels):")
            print(f"    Fmax:      {metrics['seen']['Fmax']:.4f} ★")
            print(f"    AUPR:      {metrics['seen']['aupr']:.4f}")
        
        if metrics['harmonic_mean'] is not None:
            print(f"  Harmonic Mean:   {metrics['harmonic_mean']:.4f} ★★")
    
    print(f"{'='*80}\n")