# myparser.py
from graph import Prediction, GroundTruth, propagate,Graph
import numpy as np
import logging
import re


def obo_parser(obo_file, valid_rel=("is_a", "part_of")):
    """
    Parse a OBO file and returns a list of ontologies, one for each namespace.
    Obsolete terms are excluded as well as external namespaces.
    """
    term_dict = {}
    term_id = None
    namespace = None
    name = None
    term_def = None
    alt_id = []
    rel = []
    obsolete = True
    with open(obo_file) as f:
        for line in f:
            line = line.strip().split(": ")
            if line and len(line) > 1:
                k = line[0]
                v = ": ".join(line[1:])
                if k == "id":
                    # Populate the dictionary with the previous entry
                    if term_id is not None and obsolete is False and namespace is not None:
                        term_dict.setdefault(namespace, {})[term_id] = {'name': name,
                                                                       'namespace': namespace,
                                                                       'def': term_def,
                                                                       'alt_id': alt_id,
                                                                       'rel': rel}
                    # Assign current term ID
                    term_id = v

                    # Reset optional fields
                    alt_id = []
                    rel = []
                    obsolete = False
                    namespace = None

                elif k == "alt_id":
                    alt_id.append(v)
                elif k == "name":
                    name = v
                elif k == "namespace" and v != 'external':
                    namespace = v
                elif k == "def":
                    term_def = v
                elif k == 'is_obsolete':
                    obsolete = True
                elif k == "is_a" and k in valid_rel:
                    s = v.split('!')[0].strip()
                    rel.append(s)
                elif k == "relationship" and v.startswith("part_of") and "part_of" in valid_rel:
                    s = v.split()[1].strip()
                    rel.append(s)

        # Last record
        if obsolete is False and namespace is not None:
            term_dict.setdefault(namespace, {})[term_id] = {'name': name,
                                                          'namespace': namespace,
                                                          'def': term_def,
                                                          'alt_id': alt_id,
                                                          'rel': rel}
    return term_dict


def gt_parser(gt_file, ontologies):
    """
    Parse ground truth file. Discard terms not included in the ontology.
    """
    gt_dict = {}
    with open(gt_file) as f:
        for line in f:
            line = line.strip().split()
            if line:
                p_id, term_id = line[:2]
                for ont in ontologies:
                    if term_id in ont.terms_dict:
                        gt_dict.setdefault(ont.namespace, {}).setdefault(p_id, []).append(term_id)
                        break

    gts = {}
    for ont in ontologies:
        if gt_dict.get(ont.namespace):
            matrix = np.zeros((len(gt_dict[ont.namespace]), ont.idxs), dtype='bool')
            ids = {}
            for i, p_id in enumerate(gt_dict[ont.namespace]):
                ids[p_id] = i
                for term_id in gt_dict[ont.namespace][p_id]:
                    matrix[i, ont.terms_dict[term_id]['index']] = 1
            logging.debug("gt matrix {} {} ".format(ont.namespace, matrix))
            propagate(matrix, ont, ont.order, mode='max')
            logging.debug("gt matrix propagated {} {} ".format(ont.namespace, matrix))
            gts[ont.namespace] = GroundTruth(ids, matrix, ont.namespace)
            logging.info('Ground truth: {}, proteins {}'.format(ont.namespace, len(ids)))

    return gts


def pred_parser(pred_file, ontologies, gts, prop_mode, max_terms=None):
    """
    Parse a prediction file and returns a list of prediction objects, one for each namespace.
    If a predicted is predicted multiple times for the same target, it stores the max.
    This is the slow step if the input file is huge, ca. 1 minute for 5GB input on SSD disk.

    """
    ids = {}
    matrix = {}
    ns_dict = {}  # {namespace: term}
    onts = {ont.namespace: ont for ont in ontologies}
    for ns in gts:
        matrix[ns] = np.zeros(gts[ns].matrix.shape, dtype='float')
        ids[ns] = {}
        for term in onts[ns].terms_dict:
            ns_dict[term] = ns

    with open(pred_file) as f:
        for line in f:
            line = line.strip().split()
            if line and len(line) > 2:
                p_id, term_id, prob = line[:3]
                ns = ns_dict.get(term_id)
                if ns in gts and p_id in gts[ns].ids:
                    i = gts[ns].ids[p_id]
                    if max_terms is None or np.count_nonzero(matrix[ns][i]) <= max_terms:
                        j = onts[ns].terms_dict.get(term_id)['index']
                        ids[ns][p_id] = i
                        matrix[ns][i, j] = max(matrix[ns][i, j], float(prob))

    predictions = []
    for ns in ids:
        if ids[ns]:
            logging.debug("pred matrix {} {} ".format(ns, matrix))
            propagate(matrix[ns], onts[ns], onts[ns].order, mode=prop_mode)
            logging.debug("pred matrix {} {} ".format(ns, matrix))

            predictions.append(Prediction(ids[ns], matrix[ns], len(ids[ns]), ns))
            logging.info("Prediction: {}, {}, proteins {}".format(pred_file, ns, len(ids[ns])))

    if not predictions:
        logging.warning("Empty prediction! Check format or overlap with ground truth")

    return predictions


def ia_parser(file):
    ia_dict = {}
    with open(file) as f:
        for line in f:
            if line:
                term, ia = line.strip().split()
                # term = term[3:]
                ia_dict[term] = float(ia)
    return ia_dict

def extract_go_description(term_info, name_flag='all'):
    """Extract GO description from term info"""
    if name_flag == "name":
        return term_info['name']
    
    elif name_flag == "def":
        tag_context = term_info.get('def', '')
        tag_contents = re.findall(r'"(.*?)"', tag_context)
        if tag_contents:
            return tag_contents[0]
        return ''
    
    elif name_flag == "all":
        name_part = term_info['name']
        def_part = ''
        tag_context = term_info.get('def', '')
        tag_contents = re.findall(r'"(.*?)"', tag_context)
        if tag_contents:
            def_part = tag_contents[0]
        
        if name_part and def_part:
            return f"{name_part}: {def_part}"
        elif name_part:
            return name_part
        elif def_part:
            return def_part
        else:
            return ''
    else:
        raise ValueError(f"Unknown name_flag: {name_flag}. Must be 'name', 'def', or 'all'")


def load_go_terms_and_descriptions_from_obo(obo_file, go_terms_file=None, namespace=None, 
                                            text_mode='all'):
    """Load GO terms and descriptions from OBO file"""
    print(f"\n=== Loading Ontology from {obo_file} ===")
    print(f"Text mode: {text_mode}")
    
    term_dict = obo_parser(obo_file)
    
    onto_list = []
    namespace_map = {
        'bp': 'biological_process',
        'mf': 'molecular_function',
        'cc': 'cellular_component'
    }
    
    detected_namespace = None
    user_specified_terms = False  
    
    if go_terms_file:
        user_specified_terms = True  
        print(f"\nLoading GO terms from {go_terms_file}")
        if go_terms_file.endswith('.pkl'):
            with open(go_terms_file, 'rb') as f:
                go_terms = pickle.load(f)
        else:
            with open(go_terms_file, 'r') as f:
                go_terms = [line.strip() for line in f if line.strip()]
        
        temp_onto_list = []
        for ns in term_dict:
            print(f"Loading ontology for detection: {ns}")
            ont = Graph(ns, term_dict[ns])
            temp_onto_list.append(ont)
        
        go_namespaces = {}
        for term in go_terms:
            term_with_prefix = 'GO:' + term if not term.startswith('GO:') else term
            for ont in temp_onto_list:
                if term_with_prefix in ont.terms_dict:
                    go_namespaces[term] = ont.namespace
                    break
        
        unique_namespaces = set(go_namespaces.values())
        if len(unique_namespaces) == 1:
            detected_namespace = list(unique_namespaces)[0]
            print(f"Detected ontology from GO terms: {detected_namespace}")
        elif len(unique_namespaces) > 1:
            print(f"Detected GO terms from multiple ontologies: {unique_namespaces}")
            print(f"  Will predict only the specified {len(go_terms)} GO terms")
            detected_namespace = 'mixed'  
        
        if namespace is None:
            namespace = detected_namespace
        
        onto_list = temp_onto_list
    
    if namespace and namespace != 'all' and namespace != 'mixed':
        if namespace in namespace_map:
            ns_full = namespace_map[namespace]
        else:
            ns_full = namespace
        
        onto_list = [ont for ont in onto_list if ont.namespace == ns_full]
        
        if not onto_list:
            if ns_full in term_dict:
                print(f"Loading ontology: {ns_full}")
                ont = Graph(ns_full, term_dict[ns_full])
                onto_list = [ont]
                print(f"  Terms: {ont.idxs}")
            else:
                raise ValueError(f"Namespace {ns_full} not found in OBO file")
    elif not onto_list:
        for ns in term_dict:
            print(f"Loading ontology: {ns}")
            ont = Graph(ns, term_dict[ns])
            onto_list.append(ont)
            print(f"  Terms: {ont.idxs}")
    
    if not go_terms_file:
        go_terms = []
        for ont in onto_list:
            for term_id in ont.terms_dict.keys():
                if term_id.startswith('GO:'):
                    go_terms.append(term_id.replace('GO:', ''))
        print(f"\nExtracted {len(go_terms)} GO terms from ontology")
    
    print(f"\nGenerating GO descriptions from OBO file (format: {text_mode})...")
    
    go_descriptions = []
    missing_count = 0
    
    for term in go_terms:
        term_with_prefix = 'GO:' + term if not term.startswith('GO:') else term
        description = None
        
        for ont in onto_list:
            if term_with_prefix in ont.terms_dict:
                term_info = ont.terms_dict[term_with_prefix]
                description = extract_go_description(term_info, name_flag=text_mode)
                break
        
        if description is None or description == '':
            missing_count += 1
            print(f"Warning: No description found for {term}")
            description = f"GO term {term}"
        
        go_descriptions.append(description)
    
    if missing_count > 0:
        print(f"Warning: {missing_count} terms have missing or empty descriptions")
    
    print(f"\nLoaded {len(go_terms)} GO terms with descriptions")
    print("\nExample GO descriptions:")
    for i in range(min(3, len(go_terms))):
        print(f"  {go_terms[i]}: {go_descriptions[i][:100]}{'...' if len(go_descriptions[i]) > 100 else ''}")
    
    return go_terms, go_descriptions, onto_list, detected_namespace, user_specified_terms