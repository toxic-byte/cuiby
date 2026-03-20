#main.py
from datetime import datetime
from re import S
from sklearn import preprocessing
import sys
import torch
import numpy as np
import argparse

sys.path.append(r"utils")
from dataset import obo_graph,load_datasets,process_labels_for_ontology,create_dataloaders,compute_pos_weight,create_ontology_adjacency_matrix
from config import setup_environment, get_config
from go_embed import load_nlp_model,compute_nlp_embeddings_list
from esm_embed import load_esm_model,compute_esm_embeddings
from domain_embed import load_text_pretrained_domain_features
from trainer import train_model_for_ontology
from util import filter_samples_with_labels,save_results,get_ontologies_to_train

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_mode', type=str, default='sample', 
                        choices=['full', 'sample'])
    parser.add_argument('--text_mode', type=str, default='all')
    parser.add_argument('--occ_num', type=int, default=0)
    parser.add_argument('--batch_size_train', type=int, default=16)
    parser.add_argument('--batch_size_test', type=int, default=16)
    parser.add_argument('--epoch_num', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--model', type=str, default='MZSGO')
    parser.add_argument('--nlp_model_type', type=str, default='qwen_4b')
    parser.add_argument('--esm_type', type=str, default='esm2_t33_650M_UR50D')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=1280)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--loss', type=str, default="bce")
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--onto', type=str, default="all", choices=['all', 'bp', 'mf', 'cc'])
    
    return parser.parse_args()


def main():
    args = parse_args()
    seed = setup_environment()
    config = get_config(
        run_mode=args.run_mode,
        text_mode=args.text_mode,
        occ_num=args.occ_num,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        nlp_model_type=args.nlp_model_type,
        epoch_num=args.epoch_num,
        model=args.model,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        esm_type=args.esm_type,
        embed_dim=args.embed_dim,
        loss=args.loss
    )

    ctime = datetime.now().strftime("%Y%m%d%H%M%S")
    print('Start running date:{}'.format(ctime))
    
    ontologies_to_train = get_ontologies_to_train(args.onto)
    print(f"Ontologies to train: {ontologies_to_train}")

    load_esm_model(config)
    nlp_tokenizer, nlp_model = load_nlp_model(config)
    
    label_space = {
        'biological_process': [],
        'molecular_function': [],
        'cellular_component': []
    }
    enc = preprocessing.LabelEncoder()
    
    onto, ia_dict = obo_graph(config['obo_path'],config['ia_path'])
    
    train_id, training_sequences, training_labels, test_id, test_sequences, test_labels = load_datasets(
        config, onto, label_space)
    
    train_esm_embeddings, test_esm_embeddings = compute_esm_embeddings(
        config, training_sequences, test_sequences)
    
    train_domain_features, test_domain_features = load_text_pretrained_domain_features(train_id, test_id,config)
    metrics_output_test = {}

    for key in ontologies_to_train:
        print(f"\n{'='*80}")
        print(f"Processing ontology: {key}")

        label_list, training_labels_binary, test_labels_binary, enc, ia_list,onto_parent, label_num = process_labels_for_ontology(
            config, key, label_space, training_labels, test_labels, onto, enc, ia_dict)
        
        filtered_data = filter_samples_with_labels(
            training_labels_binary, test_labels_binary,
            training_sequences, test_sequences,
            train_esm_embeddings, test_esm_embeddings,
            train_domain_features, test_domain_features,
            train_id, test_id
        )
        
        if filtered_data is None:
            print(f"  Skipping {key} - no training samples with labels")
            continue
        
        adj_matrix = create_ontology_adjacency_matrix(onto_parent, label_num, key, config)

        pos_weight = compute_pos_weight(filtered_data['train']['labels']).cuda()
        
        list_nlp = compute_nlp_embeddings_list(
            config, nlp_model, nlp_tokenizer, key, label_list, onto).cuda()
        
        train_dataloader, test_dataloader = create_dataloaders(
            config, 
            filtered_data['train']['sequences'], 
            filtered_data['train']['labels'], 
            filtered_data['train']['esm_embeddings'], 
            filtered_data['test']['sequences'], 
            filtered_data['test']['labels'], 
            filtered_data['test']['esm_embeddings'], 
            filtered_data['train']['domain_features'], 
            filtered_data['test']['domain_features']
        )
        
        model = train_model_for_ontology(
            config, key, train_dataloader, test_dataloader, list_nlp, ia_list, ctime,
            metrics_output_test, filtered_data['train']['domain_features'], adj_matrix, pos_weight,
            training_labels_binary=filtered_data['train']['labels'], 
            test_labels_binary=filtered_data['test']['labels'],       
            label_list=label_list)
    
    save_results(config, metrics_output_test, seed, ctime)
    print('End running date:{}'.format(datetime.now().strftime("%Y%m%d%H%M%S")))

if __name__ == "__main__":
    main()