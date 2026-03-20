# graph.py
import numpy as np
import copy
import logging
from collections import deque
import torch.nn.functional as F

class Graph:
    """
    Ontology class. One ontology == one namespace
    DAG is the adjacence matrix (sparse) which represent a Directed Acyclic Graph where
    DAG(i,j) == 1 means that the go term i is_a (or is part_of) j
    Parents that are in a different namespace are discarded
    """
    def __init__(self, namespace, terms_dict, ia_dict=None, orphans=False):
        """
        terms_dict = {term: {name: , namespace: , def: , alt_id: , rel:}}
        """
        self.namespace = namespace
        self.dag = []  # [[], ...] terms (rows, axis 0) x parents (columns, axis 1)
        self.terms_dict = {}  # {term: {index: , name: , namespace: , def: }  used to assign term indexes in the gt
        self.terms_list = []  # [{id: term, name:, namespace: , def:, adg: [], children: []}, ...]
        self.idxs = None  # Number of terms
        self.order = None
        self.toi = None
        self.toi_ia = None
        self.ia = None

        rel_list = []
        for self.idxs, (term_id, term) in enumerate(terms_dict.items()):
            rel_list.extend([[term_id, rel, term['namespace']] for rel in term['rel']])
            self.terms_list.append({'id': term_id, 'name': term['name'], 'namespace': namespace, 'def': term['def'],
                                 'adj': [], 'children': []})
            self.terms_dict[term_id] = {'index': self.idxs, 'name': term['name'], 'namespace': namespace, 'def': term['def']}
            for a_id in term['alt_id']:
                self.terms_dict[a_id] = copy.copy(self.terms_dict[term_id])
        self.idxs += 1

        self.dag = np.zeros((self.idxs, self.idxs), dtype='bool')

        # id1 term (row, axis 0), id2 parent (column, axis 1)
        for id1, id2, ns in rel_list:
            if self.terms_dict.get(id2):
                i = self.terms_dict[id1]['index']
                j = self.terms_dict[id2]['index']
                self.dag[i, j] = 1
                self.terms_list[i]['adj'].append(j)
                self.terms_list[j]['children'].append(i)
                logging.debug("i,j {},{} {},{}".format(i, j, id1, id2))
            else:
                logging.debug('Skipping branch to external namespace: {}'.format(id2))
        logging.debug("dag {}".format(self.dag))
        # Topological sorting
        self.top_sort()
        logging.debug("order sorted {}".format(self.order))

        if orphans:
            self.toi = np.arange(self.dag.shape[0])  # All terms, also those without parents
        else:
            self.toi = np.nonzero(self.dag.sum(axis=1) > 0)[0]  # Only terms with parents
        logging.debug("toi {}".format(self.toi))

        if ia_dict is not None:
            self.set_ia(ia_dict)

        return

    def top_sort(self):
        """
        Takes a sparse matrix representing a DAG and returns an array with nodes indexes in topological order
        https://en.wikipedia.org/wiki/Topological_sorting
        """
        indexes = []
        visited = 0
        (rows, cols) = self.dag.shape

        # create a vector containing the in-degree of each node
        in_degree = self.dag.sum(axis=0)
        # logging.debug("degree {}".format(in_degree))

        # find the nodes with in-degree 0 (leaves) and add them to the queue
        queue = np.nonzero(in_degree == 0)[0].tolist()
        # logging.debug("queue {}".format(queue))

        # for each element of the queue increment visits, add them to the list of ordered nodes
        # and decrease the in-degree of the neighbor nodes
        # and add them to the queue if they reach in-degree == 0
        while queue:
            visited += 1
            idx = queue.pop(0)
            indexes.append(idx)
            in_degree[idx] -= 1
            l = self.terms_list[idx]['adj']
            if len(l) > 0:
                for j in l:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)

        # if visited is equal to the number of nodes in the graph then the sorting is complete
        # otherwise the graph can't be sorted with topological order
        if visited == rows:
            self.order = indexes
        else:
            raise Exception("The sparse matrix doesn't represent an acyclic graph")

    def set_ia(self, ia_dict):
        self.ia = np.zeros(self.idxs, dtype='float')
        for term_id in self.terms_dict:
            if ia_dict.get(term_id):
                self.ia[self.terms_dict[term_id]['index']] = ia_dict.get(term_id)
            else:
                logging.debug('Missing IA for term: {}'.format(term_id))
        # Convert inf to zero
        np.nan_to_num(self.ia, copy=False, nan=0, posinf=0, neginf=0)
        self.toi_ia = np.nonzero(self.ia > 0)[0]


class Prediction:
    """
    The score matrix contains the scores given by the predictor for every node of the ontology
    """
    def __init__(self, ids, matrix, idx, namespace=None):
        self.ids = ids
        self.matrix = matrix  # scores
        self.next_idx = idx
        # self.n_pred_seq = idx + 1
        self.namespace = namespace

    def __str__(self):
        return "\n".join(["{}\t{}\t{}".format(index, self.matrix[index], self.namespace) for index, _id in enumerate(self.ids)])


class GroundTruth:
    def __init__(self, ids, matrix, namespace=None):
        self.ids = ids
        self.matrix = matrix
        self.namespace = namespace


def propagate(matrix, ont, order, mode='max'):
    """
    Update inplace the score matrix (proteins x terms) up to the root taking the max between children and parents
    """
    if matrix.shape[0] == 0:
        raise Exception("Empty matrix")

    deepest = np.where(np.sum(matrix[:, order], axis=0) > 0)[0][0]
    if deepest.size == 0:
        raise Exception("The matrix is empty")

    # Remove leaves
    order_ = np.delete(order, [range(0, deepest)])

    for i in order_:
        # Get direct children
        children = np.where(ont.dag[:, i] != 0)[0]
        if children.size > 0:
            # Add current terms to children
            cols = np.concatenate((children, [i]))
            if mode == 'max':
                matrix[:, i] = matrix[:, cols].max(axis=1)
            elif mode == 'fill':
                # Select only rows where the current term is 0
                rows = np.where(matrix[:, i] == 0)[0]
                if rows.size:
                    idx = np.ix_(rows, cols)
                    matrix[rows, i] = matrix[idx].max(axis=1)
    return

def check_label_propagation(labels_binary, adj_matrix, label_list, ontology_name=""):
    
    if isinstance(labels_binary, list):
        labels_binary = np.array(labels_binary)
    
    if not isinstance(labels_binary, np.ndarray):
        labels_binary = np.array(labels_binary)
    
    if isinstance(adj_matrix, torch.Tensor):
        if adj_matrix.is_sparse:
            adj_matrix = adj_matrix.to_dense().cpu().numpy()
        else:
            adj_matrix = adj_matrix.cpu().numpy()
    elif isinstance(adj_matrix, list):
        adj_matrix = np.array(adj_matrix)
    
    n_samples, n_labels = labels_binary.shape
    violations = []
    
    def get_ancestors(term_idx):
        ancestors = set()
        queue = deque([term_idx])
        
        while queue:
            current = queue.popleft()
            parents = np.where(adj_matrix[current, :] > 0)[0]
            
            for parent in parents:
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
        
        return ancestors
    
    all_ancestors = {}
    for i in range(n_labels):
        all_ancestors[i] = get_ancestors(i)
    
    if all_ancestors:
        ancestor_counts = [len(anc) for anc in all_ancestors.values()]
        avg_ancestors = np.mean(ancestor_counts)
        max_ancestors = max(ancestor_counts)
        min_ancestors = min(ancestor_counts)
       
        root_nodes = [i for i, anc in all_ancestors.items() if len(anc) == 0]

    violation_count = 0
    sample_violation_count = {}
    
    for sample_idx in range(n_samples):
        sample_labels = labels_binary[sample_idx]
        positive_labels = np.where(sample_labels == 1)[0]
        
        if len(positive_labels) == 0:
            continue
        
        for label_idx in positive_labels:
            ancestors = all_ancestors[label_idx]
            
            for ancestor_idx in ancestors:
                if sample_labels[ancestor_idx] != 1:
                    violation_count += 1
                    
                    if sample_idx not in sample_violation_count:
                        sample_violation_count[sample_idx] = 0
                    sample_violation_count[sample_idx] += 1
                    
                    violations.append({
                        'sample_idx': sample_idx,
                        'child_term': label_list[label_idx],
                        'child_idx': label_idx,
                        'missing_ancestor': label_list[ancestor_idx],
                        'ancestor_idx': ancestor_idx
                    })
    
    if violation_count == 0:
        is_propagated = True
    else:
        if sample_violation_count:
            max_violation_sample = max(sample_violation_count.items(), key=lambda x: x[1])
            violation_counts = list(sample_violation_count.values())
        
        is_propagated = False
        
    return is_propagated, violations


def propagate_labels(labels_binary, adj_matrix):
    if isinstance(labels_binary, list):
        labels_binary = np.array(labels_binary)
    
    if isinstance(adj_matrix, torch.Tensor):
        if adj_matrix.is_sparse:
            adj_matrix = adj_matrix.to_dense().cpu().numpy()
        else:
            adj_matrix = adj_matrix.cpu().numpy()
    elif isinstance(adj_matrix, list):
        adj_matrix = np.array(adj_matrix)
    
    n_samples, n_labels = labels_binary.shape
    propagated_labels = labels_binary.copy()
    
    def get_ancestors(term_idx):
        ancestors = set()
        queue = deque([term_idx])
        
        while queue:
            current = queue.popleft()
            parents = np.where(adj_matrix[current, :] > 0)[0]
            
            for parent in parents:
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
        
        return ancestors

    all_ancestors = {i: get_ancestors(i) for i in range(n_labels)}
    
    added_count = 0
    for sample_idx in range(n_samples):
        positive_labels = np.where(labels_binary[sample_idx] == 1)[0]
        
        for label_idx in positive_labels:
            ancestors = all_ancestors[label_idx]
            for ancestor_idx in ancestors:
                if propagated_labels[sample_idx, ancestor_idx] != 1:
                    propagated_labels[sample_idx, ancestor_idx] = 1
                    added_count += 1
    return propagated_labels

def enhance_embeddings_with_graph(node_features, adj_matrix, num_hops=2, alpha=0.5):
   
    h = node_features
    h_original = node_features
    
    for _ in range(num_hops):
        h = torch.sparse.mm(adj_matrix, h)
        h = F.normalize(h, p=2, dim=1)
    
    enhanced = alpha * h_original + (1 - alpha) * h
    
    return enhanced