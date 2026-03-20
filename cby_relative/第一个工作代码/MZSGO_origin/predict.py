#predict.py
import pickle
import torch
import numpy as np
import os
from tqdm import tqdm
import sys
import argparse
import json
sys.path.append(r"utils")
from config import setup_environment, get_config
from predict_util import *

def main():
    parser = argparse.ArgumentParser(description='Predict GO terms from FASTA file')
    parser.add_argument('--fasta', default="./predict_example/example.fasta", help='Input FASTA file')
    parser.add_argument('--go_terms', help='Path to GO terms list (pickle or text, optional)')
    parser.add_argument('--obo_file', default='./data/go_2023_01_01.obo', 
                       help='Path to OBO file')
    parser.add_argument('--pred_mode', choices=['all', 'bp', 'mf', 'cc'], 
                       help='Prediction mode: all, bp (biological_process), mf (molecular_function), cc (cellular_component)')
    parser.add_argument('--output', default='./predict_example/predictions.json', help='Output file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Default prediction threshold')
    parser.add_argument('--threshold_bp', type=float, default=0.29, help='Threshold for biological_process')
    parser.add_argument('--threshold_mf', type=float, default=0.46, help='Threshold for molecular_function')
    parser.add_argument('--threshold_cc', type=float, default=0.49, help='Threshold for cellular_component')
    parser.add_argument('--domain_dir', default='predict_example', help='Directory for domain files')
    parser.add_argument('--model_dir', default='./ckpt/cafa5/MZSGO')
    parser.add_argument('--text_mode', default='all', choices=['name', 'def', 'all'],
                       help='GO description format: name, def, or all (name: definition)')
    parser.add_argument('--custom_go', help='Custom GO description for prediction')
    parser.add_argument('--custom_ontology',default='cc', choices=['bp', 'mf', 'cc'])
    
    args = parser.parse_args()
    
    threshold_dict = {
        'biological_process': args.threshold_bp,
        'molecular_function': args.threshold_mf,
        'cellular_component': args.threshold_cc
    }
    
    print(f"\n=== Threshold Configuration ===")
    print(f"Biological Process (BP): {threshold_dict['biological_process']}")
    print(f"Molecular Function (MF): {threshold_dict['molecular_function']}")
    print(f"Cellular Component (CC): {threshold_dict['cellular_component']}")
    
    seed = setup_environment()
    config = get_config()
    
    namespace = None
    detected_namespace = None
    
    if args.pred_mode:
        namespace_map = {
            'bp': 'biological_process',
            'mf': 'molecular_function',
            'cc': 'cellular_component',
            'all': 'all'
        }
        namespace = namespace_map[args.pred_mode]
    
    if args.custom_go:
        if not args.custom_ontology:
            raise ValueError("--custom_ontology must be specified when using --custom_go")
        
        namespace_map = {
            'bp': 'biological_process',
            'mf': 'molecular_function',
            'cc': 'cellular_component'
        }
        custom_namespace = namespace_map[args.custom_ontology]
        
        go_terms, go_descriptions, onto_list, _, _ = load_go_terms_and_descriptions_from_obo(
            args.obo_file, 
            None,
            custom_namespace,
            args.text_mode
        )
        
        model_path = os.path.join(args.model_dir, f"{custom_namespace}.pt")
        
        print(f"\n=== Custom GO Prediction Mode ===")
        print(f"Ontology: {custom_namespace}")
        print(f"Description: {args.custom_go}")
        
        results = predict_custom_go_from_fasta(
            args.fasta,
            config,
            model_path,
            args.custom_go,
            custom_namespace,
            args.output,
            args.domain_dir
        )
        
        print("\n=== Custom GO Prediction Results ===")
        for protein_id, result in results.items():
            if 'error' in result:
                print(f"{protein_id}: ERROR - {result['error']}")
            else:
                print(f"{protein_id}: Probability = {result['probability']:.4f}")
        
        return
    
    go_terms, go_descriptions, onto_list, detected_namespace, user_specified_terms = load_go_terms_and_descriptions_from_obo(
        args.obo_file, 
        args.go_terms,
        namespace,
        args.text_mode
    )
    
    if namespace is None:
        namespace = detected_namespace
    
    if namespace is None:
        raise ValueError("Cannot determine ontology type. Please specify --pred_mode")
    
    model_file_map = {
        'biological_process': 'biological_process.pt',
        'molecular_function': 'molecular_function.pt',
        'cellular_component': 'cellular_component.pt'
    }
    
    if user_specified_terms and namespace == 'mixed':
        print("\n=== Predicting user-specified GO terms from multiple ontologies ===")
        
        terms_by_ontology = {}
        for i, term in enumerate(go_terms):
            term_with_prefix = 'GO:' + term if not term.startswith('GO:') else term
            for ont in onto_list:
                if term_with_prefix in ont.terms_dict:
                    ont_ns = ont.namespace
                    if ont_ns not in terms_by_ontology:
                        terms_by_ontology[ont_ns] = {'terms': [], 'descriptions': []}
                    terms_by_ontology[ont_ns]['terms'].append(term)
                    terms_by_ontology[ont_ns]['descriptions'].append(go_descriptions[i])
                    break
        
        all_results = {}
        
        for ont_namespace, data in terms_by_ontology.items():
            print(f"\n=== Predicting {len(data['terms'])} terms for {ont_namespace} ===")
            
            model_path = os.path.join(args.model_dir, model_file_map[ont_namespace])
            
            if not os.path.exists(model_path):
                print(f"Warning: Model not found for {ont_namespace}: {model_path}")
                continue
            
            current_threshold = threshold_dict.get(ont_namespace, args.threshold)
            print(f"Using threshold: {current_threshold:.2f}")
            
            ont_list = [ont for ont in onto_list if ont.namespace == ont_namespace]
            
            results = predict_from_fasta(
                args.fasta,
                config,
                model_path,
                data['terms'],
                data['descriptions'],
                ont_list,
                None,
                current_threshold,
                args.domain_dir,
                enable_cache=False  
            )
            
            for protein_id, result in results.items():
                if protein_id not in all_results:
                    all_results[protein_id] = {
                        'sequence_length': result['sequence_length'],
                        'predictions': [],
                        'predictions_by_ontology': {}
                    }
                all_results[protein_id]['predictions_by_ontology'][ont_namespace] = result['predictions']
                all_results[protein_id]['predictions'].extend(result['predictions'])
        
        if args.output:
            print(f"\nSaving combined results to {args.output}")
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
        
        results = all_results
    
    elif namespace == 'all':
        print("\n=== Running predictions for all ontologies ===")
        all_results = {}
        
        for ont in onto_list:
            ont_namespace = ont.namespace
            print(f"\n=== Predicting for {ont_namespace} ===")
            
            # Extract GO terms and descriptions for this ontology
            ont_go_terms = []
            ont_go_descriptions = []
            for term_id in ont.terms_dict.keys():
                if term_id.startswith('GO:'):
                    term_no_prefix = term_id.replace('GO:', '')
                    ont_go_terms.append(term_no_prefix)
                    description = extract_go_description(ont.terms_dict[term_id], name_flag=args.text_mode)
                    ont_go_descriptions.append(description)
            
            model_path = os.path.join(args.model_dir, model_file_map[ont_namespace])
            
            if not os.path.exists(model_path):
                print(f"Warning: Model not found for {ont_namespace}: {model_path}")
                continue
            
            current_threshold = threshold_dict.get(ont_namespace, args.threshold)
            print(f"Using threshold: {current_threshold:.2f}")
            
            results = predict_from_fasta(
                args.fasta,
                config,
                model_path,
                ont_go_terms,
                ont_go_descriptions,
                [ont],
                None,
                current_threshold,  
                args.domain_dir,
                enable_cache=True  
            )
            
            for protein_id, result in results.items():
                if protein_id not in all_results:
                    all_results[protein_id] = {
                        'sequence_length': result['sequence_length'],
                        'predictions': [],
                        'predictions_by_ontology': {}
                    }
                all_results[protein_id]['predictions_by_ontology'][ont_namespace] = result['predictions']
                all_results[protein_id]['predictions'].extend(result['predictions'])
        
        if args.output:
            print(f"\nSaving combined results to {args.output}")
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
        
        results = all_results
    
    else:
        # Single ontology prediction
        if namespace in model_file_map:
            model_path = os.path.join(args.model_dir, model_file_map[namespace])
            print(f"✓ Auto-selected model: {model_path}")
        elif namespace != 'all':
            model_path = os.path.join(args.model_dir, f"{namespace}.pt")
            if not os.path.exists(model_path):
                raise ValueError(f"Cannot find model for namespace: {namespace}")
        
        current_threshold = threshold_dict.get(namespace, args.threshold)
        print(f"\nUsing threshold for {namespace}: {current_threshold:.2f}")
        
        is_pred_mode = args.pred_mode is not None
        
        results = predict_from_fasta(
            args.fasta,
            config,
            model_path,
            go_terms,
            go_descriptions,
            onto_list,
            args.output,
            current_threshold,  
            args.domain_dir,
            enable_cache=is_pred_mode 
        )
    
    print("\n=== Prediction Summary ===")
    for protein_id, result in results.items():
        if 'error' in result:
            print(f"{protein_id}: ERROR - {result['error']}")
        elif 'predictions_by_ontology' in result:
            total_preds = len(result['predictions'])
            by_onto = {k: len(v) for k, v in result['predictions_by_ontology'].items()}
            print(f"{protein_id}: {total_preds} GO terms predicted ({by_onto})")
        else:
            print(f"{protein_id}: {result['num_predictions']} GO terms predicted")


if __name__ == "__main__":
    main()