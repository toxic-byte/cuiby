#test.py
from datetime import datetime
import sys
import torch
import os
import numpy as np
import argparse

sys.path.append(r"utils")
from dataset import (obo_graph, load_datasets, process_labels_for_ontology,
                     create_dataloaders, create_ontology_adjacency_matrix)
from config import setup_environment, get_config
from go_embed import load_nlp_model,compute_nlp_embeddings_list
from esm_embed import (load_esm_model,compute_esm_embeddings)
from domain_embed import load_text_pretrained_domain_features
from model import CustomModel
from sklearn import preprocessing
import torch.nn as nn
from trainer import evaluate_model_with_unseen
from test_zero import identify_unseen_labels,print_unseen_label_analysis,save_test_results
from util import get_ontologies_to_train

def parse_args():
    parser = argparse.ArgumentParser(description='Test trained models for protein function prediction')
    
    parser.add_argument('--run_mode', type=str, default='sample', 
                        choices=['full','sample','zero'])
    parser.add_argument('--text_mode', type=str, default='all')
    parser.add_argument('--occ_num', type=int, default=0)
    parser.add_argument('--batch_size_train', type=int, default=16)
    parser.add_argument('--batch_size_test', type=int, default=16)
    parser.add_argument('--nlp_model_type', type=str, default='qwen_4b')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--onto', type=str, default="all", 
                        choices=['all', 'bp', 'mf', 'cc'],
                        help='Specify which ontology to test: all/bp/mf/cc')
    
    parser.add_argument('--bp_model_path', type=str, 
                        default='./ckpt/cafa5/MZSGO/biological_process.pt',
                        help='Path to biological_process model checkpoint')
    parser.add_argument('--cc_model_path', type=str, 
                        default='./ckpt/cafa5/MZSGO/cellular_component.pt',
                        help='Path to cellular_component model checkpoint')
    parser.add_argument('--mf_model_path', type=str, 
                        default='./ckpt/cafa5/MZSGO/molecular_function.pt',
                        help='Path to molecular_function model checkpoint')
    
    return parser.parse_args()

def load_trained_model(checkpoint_path, config, train_domain_features):
    print(f"\nLoading model from: {checkpoint_path}")
    
    model = CustomModel(
        esm_dim=config['embed_dim'],
        nlp_dim=config['nlp_dim'],
        domain_size=train_domain_features.shape[1],
        hidden_dim=config.get('hidden_dim', 512),
    ).cuda()
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint)    
    return model, checkpoint

def main_test():
    args = parse_args()
    seed = setup_environment()
    config = get_config(
        run_mode=args.run_mode, 
        text_mode=args.text_mode, 
        occ_num=args.occ_num,
        batch_size_train=args.batch_size_train, 
        batch_size_test=args.batch_size_test,
        nlp_model_type=args.nlp_model_type, 
        hidden_dim=args.hidden_dim
    )
    
    print('='*80)
    print('Start testing at: {}'.format(datetime.now().strftime("%Y%m%d%H%M%S")))
    
    ontologies_to_test = get_ontologies_to_train(args.onto)
    print(f"Ontologies to test: {ontologies_to_test}")
    
    model_paths = {
        'biological_process': args.bp_model_path,
        'cellular_component': args.cc_model_path,
        'molecular_function': args.mf_model_path
    }
    
    load_esm_model(config)
    nlp_tokenizer, nlp_model = load_nlp_model(config)
 
    label_space = {
        'molecular_function': [],
        'biological_process': [],
        'cellular_component': []
    }
    enc = preprocessing.LabelEncoder()
    
    onto, ia_dict = obo_graph(config['obo_path'], config['ia_path'])
    
    train_id, training_sequences, training_labels, test_id, test_sequences, test_labels = load_datasets(
        config, onto, label_space)
    
    _, test_esm_embeddings = compute_esm_embeddings(
        config, training_sequences, test_sequences)
    
    train_domain_features,test_domain_features=load_text_pretrained_domain_features(train_id,test_id,config)
    
    all_results = {}
    
    for key in ontologies_to_test:
        print(f"\n{'='*80}")
        print(f"Testing for ontology: {key}")
        
        checkpoint_path = model_paths[key]
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Model file not found: {checkpoint_path}")
            print(f"Skipping {key}")
            continue
        
        label_list, training_labels_binary, test_labels_binary, enc, ia_list, onto_parent, label_num = process_labels_for_ontology(
            config, key, label_space, training_labels, test_labels, onto, enc, ia_dict)
        
        unseen_indices, seen_indices, train_counts, test_counts = identify_unseen_labels(
            training_labels_binary, test_labels_binary)
        
        print_unseen_label_analysis(key, unseen_indices, seen_indices, 
                                    train_counts, test_counts, label_list)
        
        adj_matrix = create_ontology_adjacency_matrix(onto_parent, label_num, key, config)
        
        list_nlp = compute_nlp_embeddings_list(
            config, nlp_model, nlp_tokenizer, key, label_list, onto).cuda()
                
        _, test_dataloader = create_dataloaders(
            config, training_sequences, training_labels_binary, _,
            test_sequences, test_labels_binary, test_esm_embeddings,
            train_domain_features, test_domain_features)
        
        model, checkpoint = load_trained_model(checkpoint_path, config, train_domain_features)
        
        metrics = evaluate_model_with_unseen(
            model, test_dataloader, list_nlp, ia_list, key, adj_matrix,
            unseen_indices, seen_indices)
        
        all_results[key] = {
            'metrics': metrics,
            'checkpoint_path': checkpoint_path,
            'unseen_count': len(unseen_indices),
            'seen_count': len(seen_indices)
        }
    
    if all_results:
        save_test_results(config, all_results)
        
        print(f"\n{'='*80}")
        print(f"TEST RESULTS SUMMARY")
        print(f"{'='*80}")
        for key, result in all_results.items():
            metrics = result['metrics']
            print(f"\n{key}:")
            print(f"  Overall Metrics:")
            print(f"    Avg Fmax:      {metrics['Fmax']:.4f}★")
            print(f"    Prop-Fmax:     {metrics['prop_Fmax']:.4f} ")
            print(f"    AUPR:          {metrics['aupr']:.4f}")
            print(f"    Prop-AUPR:     {metrics['prop_aupr']:.4f}")
            
            if metrics['unseen'] is not None:
                print(f"  Unseen Labels ({result['unseen_count']} labels):")
                print(f"    Avg Fmax:      {metrics['unseen']['Fmax']:.4f}")
                if 'prop_Fmax' in metrics['unseen']:
                    print(f"    Prop-Fmax:     {metrics['unseen']['prop_Fmax']:.4f}")
                print(f"    AUPR:          {metrics['unseen']['aupr']:.4f} ★")
                if 'prop_aupr' in metrics['unseen']:
                    print(f"    Prop-AUPR:     {metrics['unseen']['prop_aupr']:.4f}")
            
            if metrics['seen'] is not None:
                print(f"  Seen Labels ({result['seen_count']} labels):")
                print(f"    Avg Fmax:      {metrics['seen']['Fmax']:.4f}")
                if 'prop_Fmax' in metrics['seen']:
                    print(f"    Prop-Fmax:     {metrics['seen']['prop_Fmax']:.4f}")
                print(f"    AUPR:          {metrics['seen']['aupr']:.4f} ★")
                if 'prop_aupr' in metrics['seen']:
                    print(f"    Prop-AUPR:     {metrics['seen']['prop_aupr']:.4f}")
            
            if metrics['harmonic_mean'] is not None:
                print(f"  Harmonic Mean:   {metrics['harmonic_mean']:.4f} ★★")
        
        print(f"{'='*80}\n")
    else:
        print("\nNo results to display. Please check model paths and ontology selection.")
    
    print('End testing at: {}'.format(datetime.now().strftime("%Y%m%d%H%M%S")))


if __name__ == "__main__":
    main_test()