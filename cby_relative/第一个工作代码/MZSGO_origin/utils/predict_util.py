#predict_util.py
import pickle
import torch
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import json
import ssl
from urllib import request
from urllib.error import HTTPError
from time import sleep
from go_embed import load_nlp_model,encode_go_descriptions_batch,encode_go_description_single
from esm_embed import load_esm_model,compute_esm_embeddings_single
from myparser import extract_go_description,load_go_terms_and_descriptions_from_obo
from domain_embed import load_domain_embeddings,encode_domain_features_by_list,encode_domain_features_by_protein_id


class ProteinGOPredictor:
    def __init__(self, config, model_path, go_terms, go_descriptions=None, onto=None, enable_cache=False):
       
        self.config = config
        self.go_terms = go_terms
        self.num_labels = len(go_terms)
        self.onto = onto
        self.go_descriptions = go_descriptions
        self.enable_cache = enable_cache
        
        print("Initializing predictor...")
        load_esm_model(config)
        load_nlp_model(config)
        load_domain_embeddings(config)
        
        self.go_embeddings = self._load_or_compute_go_embeddings()
        print(f"GO embeddings shape: {self.go_embeddings.shape}")
        
        self._load_model(model_path)
    
    def _load_or_compute_go_embeddings(self):
        """Load cached GO embeddings or compute them (cache only if enable_cache=True)"""
        cache_dir = './data/embeddings_cache/onto'
        os.makedirs(cache_dir, exist_ok=True)
        
        ontology_type = 'unknown'
        if self.onto and len(self.onto) > 0:
            ontology_type = self.onto[0].namespace
        
        text_mode = self.config.get('text_mode', 'all')
        nlp_name = self.config.get('nlp_name', 'Qwen/Qwen3-Embedding-4B')
        model_short_name = nlp_name.split('/')[-1]
        
        cache_filename = f"{ontology_type}_{text_mode}_{model_short_name}.pkl"
        cache_path = os.path.join(cache_dir, cache_filename)
        
        if self.enable_cache and os.path.exists(cache_path):
            print(f"Loading cached GO embeddings from {cache_path}...")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if cached_data['go_terms'] == self.go_terms:
                    print(f"✓ Cache hit! Loaded {len(self.go_terms)} GO embeddings")
                    return torch.tensor(cached_data['embeddings'], dtype=torch.float32)
                else:
                    print("⚠ Cache mismatch, recomputing embeddings...")
            except Exception as e:
                print(f"⚠ Failed to load cache: {e}, recomputing embeddings...")
        
        if self.enable_cache:
            print("Pre-computing GO description embeddings (will be cached)...")
        else:
            print("Pre-computing GO description embeddings (no caching)...")
        
        go_embeddings = encode_go_descriptions_batch(self.go_descriptions, self.config)
        
        if self.enable_cache:
            print(f"Saving GO embeddings to cache: {cache_path}")
            cache_data = {
                'go_terms': self.go_terms,
                'go_descriptions': self.go_descriptions,
                'embeddings': go_embeddings.numpy(),
                'ontology_type': ontology_type,
                'text_mode': text_mode,
                'nlp_model': nlp_name
            }
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print("Cache saved successfully")
            except Exception as e:
                print(f"Failed to save cache: {e}")
        else:
            print("✓ Embeddings computed (caching disabled for this run)")
        
        return go_embeddings
    
    def _load_model(self, model_path):
        from model import CustomModel
        
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, weights_only=False)
        
        esm_dim = self.config.get('embed_dim', 1280)
        nlp_dim = self.config.get('nlp_dim', 2560)
        
        domain_embeddings_dict = load_domain_embeddings(self.config)
        if domain_embeddings_dict:
            sample_key = list(domain_embeddings_dict.keys())[0]
            sample_value = domain_embeddings_dict[sample_key]
            if isinstance(sample_value, dict) and 'embedding' in sample_value:
                domain_size = sample_value['embedding'].shape[0]
            else:
                domain_size = nlp_dim
        else:
            domain_size = nlp_dim
        
        hidden_dim = self.config.get('hidden_dim', 512)
        
        self.model = CustomModel(
            esm_dim=esm_dim,
            nlp_dim=nlp_dim,
            domain_size=domain_size,
            hidden_dim=hidden_dim,
        )
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.cuda()
        self.model.eval()
        print("Model loaded successfully")
    
    def predict_single(self, sequence, domain_input, threshold=0.5):
        esm_embedding = compute_esm_embeddings_single(sequence, self.config)
        
        if isinstance(domain_input, str):
            domain_embedding = encode_domain_features_by_protein_id(domain_input, self.config)
        elif isinstance(domain_input, list):
            domain_embedding = encode_domain_features_by_list(domain_input, self.config)
        else:
            raise ValueError("domain_input must be protein_id (str) or domain_list (list)")
        
        batch_size = 1
        num_labels = self.num_labels
        
        esm_embedding = esm_embedding.unsqueeze(0).expand(num_labels, -1)
        domain_embedding = domain_embedding.unsqueeze(0).expand(num_labels, -1)
        go_embeddings = self.go_embeddings
        
        with torch.no_grad():
            esm_embedding = esm_embedding.cuda()
            domain_embedding = domain_embedding.cuda()
            go_embeddings = go_embeddings.cuda()
            
            logits = self.model(
                esm_embedding=esm_embedding,
                domain_embedding=domain_embedding,
                nlp_embedding=go_embeddings,
                batch_size=batch_size
            )
            
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        predictions = []
        for i, (term_id, prob) in enumerate(zip(self.go_terms, probs)):
            if prob >= threshold:
                predictions.append({
                    'go_term': term_id,
                    'description': self.go_descriptions[i],
                    'probability': float(prob)
                })
        
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'sequence_length': len(sequence),
            'num_predictions': len(predictions),
            'predictions': predictions
        }
    
    def predict_with_custom_go(self, sequence, domain_input, go_description):
        esm_embedding = compute_esm_embeddings_single(sequence, self.config)
        
        if isinstance(domain_input, str):
            domain_embedding = encode_domain_features_by_protein_id(domain_input, self.config)
        elif isinstance(domain_input, list):
            domain_embedding = encode_domain_features_by_list(domain_input, self.config)
        else:
            raise ValueError("domain_input must be protein_id (str) or domain_list (list)")
        
        go_embedding = encode_go_description_single(go_description, self.config)
        
        esm_embedding = esm_embedding.unsqueeze(0)
        domain_embedding = domain_embedding.unsqueeze(0)
        go_embedding = go_embedding.unsqueeze(0)
        
        with torch.no_grad():
            esm_embedding = esm_embedding.cuda()
            domain_embedding = domain_embedding.cuda()
            go_embedding = go_embedding.cuda()
            
            logits = self.model(
                esm_embedding=esm_embedding,
                domain_embedding=domain_embedding,
                nlp_embedding=go_embedding,
                batch_size=1
            )
            
            prob = torch.sigmoid(logits).cpu().item()
        
        return prob


def predict_from_fasta(fasta_file, config, model_path, go_terms, go_descriptions, 
                       onto_list, output_file=None, threshold=0.5, domain_dir="predict_example",
                       enable_cache=False):
    print(f"Parsing FASTA file: {fasta_file}")
    sequences = parse_fasta(fasta_file)
    print(f"Found {len(sequences)} sequences")
    
    print("\nDownloading domain information from InterPro...")
    protein_domains = {}
    for protein_id in tqdm(sequences.keys(), desc="Downloading domains"):
        domains = fetch_interpro_domains_for_protein(protein_id, domain_dir)
        protein_domains[protein_id] = domains
    
    print("\nInitializing predictor...")
    predictor = ProteinGOPredictor(config, model_path, go_terms, go_descriptions, 
                                   onto_list, enable_cache=enable_cache)
    
    print("\nPredicting GO terms...")
    results = {}
    for protein_id, sequence in tqdm(sequences.items(), desc="Predicting"):
        domains = protein_domains[protein_id]
        
        try:
            result = predictor.predict_single(sequence, domains, threshold=threshold)
            results[protein_id] = result
        except Exception as e:
            print(f"Error predicting {protein_id}: {e}")
            results[protein_id] = {
                'error': str(e),
                'sequence_length': len(sequence),
                'num_predictions': 0,
                'predictions': []
            }
    
    if output_file:
        print(f"\nSaving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def predict_custom_go_from_fasta(fasta_file, config, model_path, go_description, 
                                 ontology_type, output_file=None, domain_dir="predict_example"):
    print(f"Parsing FASTA file: {fasta_file}")
    sequences = parse_fasta(fasta_file)
    print(f"Found {len(sequences)} sequences")
    
    print("\nDownloading domain information from InterPro...")
    protein_domains = {}
    for protein_id in tqdm(sequences.keys(), desc="Downloading domains"):
        domains = fetch_interpro_domains_for_protein(protein_id, domain_dir)
        protein_domains[protein_id] = domains
    
    print("\nInitializing predictor...")
    dummy_go_terms = ['0000000']
    dummy_go_descriptions = [go_description]
    
    predictor = ProteinGOPredictor(config, model_path, dummy_go_terms, 
                                   dummy_go_descriptions, None, enable_cache=False)
    
    print("\nPredicting custom GO term...")
    results = {}
    for protein_id, sequence in tqdm(sequences.items(), desc="Predicting"):
        domains = protein_domains[protein_id]
        
        try:
            prob = predictor.predict_with_custom_go(sequence, domains, go_description)
            results[protein_id] = {
                'sequence_length': len(sequence),
                'probability': float(prob),
                'go_description': go_description,
                'ontology_type': ontology_type
            }
        except Exception as e:
            print(f"Error predicting {protein_id}: {e}")
            results[protein_id] = {
                'error': str(e),
                'sequence_length': len(sequence)
            }
    
    if output_file:
        print(f"\nSaving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def parse_fasta(fasta_file):
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def fetch_interpro_domains_for_protein(protein_id, output_dir="predict_example", max_retries=3):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{protein_id}.tsv")
    
    if os.path.exists(file_path):
        return parse_domain_file(file_path)
    
    url = f"https://www.ebi.ac.uk/interpro/api/entry/InterPro/protein/reviewed/{protein_id}/?page_size=200"
    context = ssl._create_unverified_context()
    
    attempts = 0
    while attempts < max_retries:
        try:
            req = request.Request(url, headers={"Accept": "application/json"})
            res = request.urlopen(req, context=context, timeout=60)
            
            if res.status == 408:
                sleep(61)
                continue
            elif res.status == 204:
                with open(file_path, "w") as f:
                    pass
                return []
            
            payload = json.loads(res.read().decode())
            
            domains = []
            with open(file_path, "w") as f:
                for item in payload.get("results", []):
                    metadata = item.get("metadata", {})
                    accession = metadata.get("accession", "")
                    if accession:
                        domains.append(accession)
                        f.write(f"{accession}\n")
            
            return domains
            
        except HTTPError as e:
            if e.code == 408:
                sleep(61)
                attempts += 1
                continue
            elif e.code == 404:
                with open(file_path, "w") as f:
                    pass
                return []
            else:
                attempts += 1
                if attempts < max_retries:
                    sleep(10)
                    continue
                else:
                    print(f"Failed to fetch domains for {protein_id}: {e}")
                    return []
        except Exception as e:
            attempts += 1
            if attempts < max_retries:
                sleep(10)
                continue
            else:
                print(f"Error fetching domains for {protein_id}: {e}")
                return []
    
    return []


def parse_domain_file(domain_file_path):
    domains = []
    
    if not os.path.exists(domain_file_path):
        return domains
    
    with open(domain_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 1:
                domain_id = parts[0]
                if domain_id:
                    domains.append(domain_id)
    
    return domains