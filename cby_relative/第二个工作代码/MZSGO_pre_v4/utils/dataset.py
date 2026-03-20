import torch
from myparser import obo_parser, ia_parser
from graph import Graph
from torch.utils.data import Dataset,DataLoader
import os
import pickle
from collections import Counter
from tqdm import tqdm
import numpy as np

# Parse the OBO file and creates a different graph for each namespace
def obo_graph(filepath, dict_path=None):
    ia_dict = None
    if dict_path is not None:
        ia_dict = ia_parser(dict_path)

    ontologies = []
    no_orphans = False
    for ns, terms_dict in obo_parser(filepath).items():
        ontologies.append(Graph(ns, terms_dict, ia_dict, not no_orphans))
    return ontologies, ia_dict

def parent(enc, key, label_list,onto,label_space):
    onto_parent = {}
    label_num = len(enc.classes_)
    for i in range(label_num):
        _label = enc.inverse_transform([i])
        _tag = 'GO:' + str(_label[0])
        if i not in onto_parent.keys():
            onto_parent[i] = {
                'size': 0,
                'pos': []
            }
        for ont in onto:
            if ont.namespace != key:
                continue
            for term in ont.terms_list:
                if term['id'] == _tag:
                    ns = ont.namespace
                    parent_ids = term['adj']
                    if len(parent_ids) == 0:
                        continue
                    else:
                        for _parent in parent_ids:
                            for _key, val in ont.terms_dict.items():
                                if 'index' in val and val['index'] == _parent:
                                    poss_tags = _key[3:]
                                    if poss_tags not in label_space[
                                        key]:  # 'alt_id' is used in this version, has to exclude from ground-truth label space
                                        continue
                                    if poss_tags not in label_list:
                                        continue
                                    _pos = enc.transform([poss_tags])
                                    onto_parent[i]['size'] += 1
                                    onto_parent[i]['pos'].extend(_pos)
    return onto_parent

def preprocess_dataset(filepath, MAXLEN,onto,label_space):
    '''
        Args:
            sequences: list, the list which contains the protein primary sequences.
            labels: list, the list which contains the dataset labels.
            max_length, Integer, the maximum sequence length,
            if there is a sequence that is larger than the specified sequence length will be post-truncated.
    '''
    pro_id = []
    sequences = []
    labels = {
        'biological_process': [],
        'molecular_function': [],
        'cellular_component': []
    }
    multi_labels = {
        'biological_process': [],
        'molecular_function': [],
        'cellular_component': []
    }
    path = filepath
    print(f"Start reading {path}")
    with open(path, "r") as lines:
        for _line in lines:
            if _line.startswith('>'):
                _line = _line.strip()
                seqs = _line.split()
                _id = seqs[0][1:]
                pro_id.append(_id)
                tags = seqs[1].split(';')
                for tag in tags:
                    gene = 'GO:' + tag
                    for ont in onto:
                        ns = ont.namespace
                        if gene in ont.terms_dict.keys():
                            multi_labels[ns].append(tag)
                            label_space[ns].append(tag)
                            continue
                for key in multi_labels.keys():
                    labels[key].append(multi_labels[key])
                multi_labels = {
                    'biological_process': [],
                    'molecular_function': [],
                    'cellular_component': []
                }
            else:
                _line = _line.strip()
                if len(_line) > MAXLEN:
                    _line = _line[:MAXLEN]
                sequences.append(_line)
    print("Read input complete")
    return pro_id, sequences, labels

class StabilitylandscapeDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, idx):
        embedding = self.sequences[idx]
        label = self.labels[idx]
        return {'embed': embedding, 'labels': torch.as_tensor(label, dtype=torch.float32).clone().detach()}

    def __len__(self):
        return len(self.sequences)
    
class IndexedStabilitylandscapeDataset(StabilitylandscapeDataset):
    def __init__(self, sequences, labels, embeddings=None, domain_features=None):
        super().__init__(sequences, labels)
        self.embeddings = embeddings
        self.domain_features=domain_features 
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data['index'] = idx  
        
        if self.embeddings is not None:
            data['embedding'] = self.embeddings[idx]  
        
        if self.domain_features is not None:
            data['domain_feature'] = torch.FloatTensor(self.domain_features[idx])  
        return data
    
    def __len__(self):
        return len(self.sequences)

def load_datasets(config, onto, label_space):
    print("Loading datasets...")
    train_id, training_sequences, training_labels = preprocess_dataset(
        config['train_path'], config['MAXLEN'], onto, label_space
    )
    test_id, test_sequences, test_labels = preprocess_dataset(
        config['test_path'], config['MAXLEN'], onto, label_space
    )
    
    print("Train IDs (first 5):", train_id[:5])
    print("Test IDs (first 5):", test_id[:5])
    print(f"Total train samples: {len(train_id)}, Total test samples: {len(test_id)}")
    
    return train_id, training_sequences, training_labels, test_id, test_sequences, test_labels


def process_labels_for_ontology(config, key, label_space, training_labels, test_labels, onto, enc, ia_dict):
    print(f"\n{'='*50}")
    print(f"Processing labels for ontology: {key}")
    
    if config['run_mode'] == "sample":
        label_processing_cache = os.path.join(config['cache_dir'], f"labels/{config['occ_num']}/label_processed_{key}_sample.pkl")
    elif config['run_mode'] == "full":
        label_processing_cache = os.path.join(config['cache_dir'], f"labels/{config['occ_num']}/label_processed_{key}.pkl")
    elif config['run_mode'] == "zero":
        label_processing_cache = os.path.join(config['cache_dir'], f"labels/{config['occ_num']}/label_processed_{key}_zero.pkl")
    label_processing_dir = os.path.dirname(label_processing_cache)
    if label_processing_dir and not os.path.exists(label_processing_dir):
        os.makedirs(label_processing_dir, exist_ok=True)

    if os.path.exists(label_processing_cache):
        print(f"Loading preprocessed labels for {key} from cache...")
        with open(label_processing_cache, 'rb') as f:
            cached = pickle.load(f)
        
        return (cached['label_list'], cached['training_labels_binary'], 
                cached['test_labels_binary'], cached['encoder'], 
                cached['ia_list'], cached['onto_parent'], cached['label_num'])
    
    print(f"Processing labels for {key} from scratch...")
    
    label_tops = Counter(label_space[key])
    top_labels = sorted([label for label in set(label_space[key]) if label_tops[label] > config['occ_num']])
    print(f'Top label numbers: {len(top_labels)}')
    label_list = top_labels
    print("Top labels (first 10):", label_list[:10])
    
    labspace = enc.fit_transform(label_list)
    onto_parent = parent(enc, key, label_list, onto, label_space)
    label_num = len(enc.classes_)
    print(f'Number of classes: {label_num}')
    
    label_set = set(label_list)
    training_labels_binary = convert_labels_to_binary(training_labels[key], label_set, enc, label_num)
    test_labels_binary = convert_labels_to_binary(test_labels[key], label_set, enc, label_num)
    
    ia_list = build_ia_weight_matrix(ia_dict, label_set, enc, label_num)
    
    print(f"Saving processed labels to {label_processing_cache}")
    with open(label_processing_cache, 'wb') as f:
        pickle.dump({
            'label_list': label_list,
            'training_labels_binary': training_labels_binary,
            'test_labels_binary': test_labels_binary,
            'encoder': enc,
            'ia_list': ia_list,
            'onto_parent': onto_parent,
            'label_num': label_num,
        }, f)
    print("✓ Saved processed labels")
    
    return label_list, training_labels_binary, test_labels_binary, enc, ia_list, onto_parent, label_num

def convert_labels_to_binary(labels, label_set, enc, label_num):
    print("Converting labels to binary format...")
    labels_binary = []
    for label in tqdm(labels, desc="Processing labels"):
        filtered_label = [item for item in label if item in label_set]
        if len(filtered_label) == 0:
            labels_binary.append([0] * label_num)
        else:
            temp_labels = enc.transform(filtered_label)
            binary_label = [0] * label_num
            for idx in temp_labels:
                binary_label[idx] = 1
            labels_binary.append(binary_label)
    return labels_binary

def to_label_tensor(y):
    if isinstance(y, torch.Tensor):
        if y.dtype != torch.float32:
            y = y.float()
        return y.detach().cpu()
    elif isinstance(y, np.ndarray):
        return torch.from_numpy(y.astype(np.float32))
    elif isinstance(y, list):
        if len(y) == 0:
            raise ValueError("Empty label list.")
        return torch.tensor(y, dtype=torch.float32)
    else:
        raise TypeError(f"Unsupported type for labels: {type(y)}")

def compute_pos_weight(y, smoothing=1.0, clip_min=1.0, clip_max=10.0,
                       use_log_compress=True, device=None):
    y = to_label_tensor(y)   # CPU float32 tensor [N, L]
    N = y.size(0)
    pos = y.sum(dim=0)       # [L]
    neg = N - pos            # [L]

    if use_log_compress:
        ratio = (neg + smoothing) / (pos + smoothing)
        pw = 1.0 + torch.log(ratio.clamp(min=1e-8))
    else:
        pw = (neg + smoothing) / (pos + smoothing)

    if clip_min is not None or clip_max is not None:
        if clip_min is None: clip_min = float('-inf')
        if clip_max is None: clip_max = float('inf')
        pw = pw.clamp(min=clip_min, max=clip_max)

    if device is not None:
        pw = pw.to(device)
    return pw  # [L] float tensor

def build_ia_weight_matrix(ia_dict, label_set, enc, label_num):
    print("Building IA weight matrix...")
    ia_list = torch.ones(1, label_num).cuda()
    for _tag, _value in ia_dict.items():
        _tag = _tag[3:]
        if _tag not in label_set:
            continue
        ia_id = enc.transform([_tag])
        if _value == 0.0:
            _value = 1.0
        ia_list[0, ia_id[0]] = _value
    return ia_list

def create_ontology_adjacency_matrix(onto_parent, label_num, key,config):
    cache_path=f"./data/embeddings_cache/adj/{config['occ_num']}/adj_matrix_{key}_{config['run_mode']}.pt"
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Loading adjacency matrix from cache: {cache_path}")
        try:
            adj_matrix = torch.load(cache_path)
            print("Successfully loaded cached adjacency matrix")
            return adj_matrix
        except Exception as e:
            print(f"Failed to load cache: {e}, regenerating...")
    
    print("Generating new adjacency matrix...")
    adj_matrix = torch.zeros(label_num, label_num).cuda()
    
    for i in range(label_num):
        position = onto_parent[i]['pos'].copy()
        adj_matrix[i, i] = 1.0 
        for j in position:
            adj_matrix[i, j] = 1.0  
    
    sparse_adj_matrix = adj_matrix.to_sparse()
    
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            torch.save(sparse_adj_matrix, cache_path)
            print(f"Adjacency matrix saved to cache: {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    return sparse_adj_matrix

def create_dataloaders(config, training_sequences, training_labels_binary, train_esm_embeddings,
                       test_sequences, test_labels_binary, test_esm_embeddings,train_domain_features, test_domain_features):
    training_dataset = IndexedStabilitylandscapeDataset(
        training_sequences, 
        training_labels_binary, 
        embeddings=train_esm_embeddings,
        domain_features=train_domain_features

    )
    test_dataset = IndexedStabilitylandscapeDataset(
        test_sequences, 
        test_labels_binary, 
        embeddings=test_esm_embeddings,
        domain_features=test_domain_features
    )
    
    train_dataloader = DataLoader(
        training_dataset, 
        batch_size=config['batch_size_train'], 
        shuffle=True,
        num_workers=config.get('num_workers', 0),  
        pin_memory=False  
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size_test'], 
        shuffle=False,
        num_workers=config.get('num_workers', 0),  
        pin_memory=False  
    )
    
    return train_dataloader, test_dataloader

