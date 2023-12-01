import torch
from rdkit.Chem import MolFromSmiles
import numpy as np
from rdkit import Chem
from SCI.Featurizer import *
import pickle
import time


smilesList = ['CC']
degrees = [0, 1, 2, 3, 4, 5]


class MolGraph(object):
    def __init__(self):
        self.nodes = {} # dict of lists of nodes, keyed by node type

    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]

class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]



def graph_from_smiles_tuple(smiles_tuple):
    graph_list = [graph_from_smiles(s) for s in smiles_tuple]
    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)

    # This sorting allows an efficient (but brittle!) indexing later on.
    big_graph.sort_nodes_by_degree('atom')
    return big_graph
def bond_index_from_smiles(smiles,max_atom):
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    bond_index_plus = torch.empty(2, len(mol.GetBonds())).to('cuda')

    i = 0
    for bond in mol.GetBonds():
        bond_index_plus[0][i] = bond.GetBeginAtom().GetIdx()
        bond_index_plus[1][i] = bond.GetEndAtom().GetIdx()
        i=i+1
    bond_index_plus = bond_index_plus.long()
    v = torch.ones(bond_index_plus.size(1)).float()
    adj_sparse = torch.sparse.FloatTensor(bond_index_plus.to('cuda'), v.to('cuda'),
                                          torch.Size([max_atom.shape[0], max_atom.shape[0]]))
    adj_dense = adj_sparse.to_dense()
    return adj_dense

def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = MolFromSmiles(smiles)
    atom_index = []
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        index = atom.GetAtomicNum()
        atom_index.append(index)
        new_atom_node = graph.new_node('atom', features = atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node
    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features = bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))
    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph,atom_index

def array_rep_from_smiles(molgraph,atom_index):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    degrees = [0,1,2,3,4,5]
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'),
                'rdkit_ix'      : molgraph.rdkit_ix_array(),
                'atom_index': atom_index}

    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))

def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


def gen_descriptor_data(smilesList):

    smiles_to_fingerprint_array = {}

    for i,smiles in enumerate(smilesList):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        try:
            molgraph,atom_index = graph_from_smiles(smiles)
            molgraph.sort_nodes_by_degree('atom')
            arrayrep = array_rep_from_smiles(molgraph,atom_index)

            smiles_to_fingerprint_array[smiles] = arrayrep
            #smiles_to_fingerprint_array[smiles] = atom_index
        except:
            print(smiles)
            time.sleep(3)
    return smiles_to_fingerprint_array

#-----------------------------------------------------------------------------------------------
def save_smiles_dicts(smilesList,filename):
    #first need to get the max atom length
    max_atom_len = 0
    max_bond_len = 0
    num_atom_features = 0
    num_bond_features = 0
    smiles_to_rdkit_list = {}
    smiles_to_fingerprint_features = gen_descriptor_data(smilesList)
    for smiles,arrayrep in smiles_to_fingerprint_features.items():
        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        rdkit_list = arrayrep['rdkit_ix']
        smiles_to_rdkit_list[smiles] = rdkit_list

        atom_len,num_atom_features = atom_features.shape
        bond_len,num_bond_features = bond_features.shape

        if atom_len > max_atom_len:
            max_atom_len = atom_len
        if bond_len > max_bond_len:
            max_bond_len = bond_len

    #then add 1 so I can zero pad everything
    max_atom_index_num = max_atom_len
    max_bond_index_num = max_bond_len
    max_atom_len += 1
    max_bond_len += 1
    smiles_to_atom_info = {}
    smiles_to_bond_info = {}
    smiles_to_atom_neighbors = {}
    smiles_to_bond_neighbors = {}
    smiles_to_atom_index={}
    smiles_to_atom_mask = {}
    degrees = [0,1,2,3,4,5]
    #then run through our numpy array again
    for smiles,arrayrep in smiles_to_fingerprint_features.items():
        mask = np.zeros((max_atom_len))
        #get the basic info of what
        #    my atoms and bonds are initialized
        atoms = np.zeros((max_atom_len,num_atom_features))
        bonds = np.zeros((max_bond_len,num_bond_features))

        #then get the arrays initlialized for the neighbors
        atom_neighbors = np.zeros((max_atom_len,len(degrees)))
        bond_neighbors = np.zeros((max_atom_len,len(degrees)))

        #now set these all to the last element of the list, which is zero padded
        atom_neighbors.fill(max_atom_index_num)
        bond_neighbors.fill(max_bond_index_num)
        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']
        for i,feature in enumerate(atom_features):
            mask[i] = 1.0
            atoms[i] = feature

        for j,feature in enumerate(bond_features):
            bonds[j] = feature

        atom_neighbor_count = 0
        bond_neighbor_count = 0
        for degree in degrees:
            atom_neighbors_list = arrayrep[('atom_neighbors', degree)]
            bond_neighbors_list = arrayrep[('bond_neighbors', degree)]

            if len(atom_neighbors_list) > 0:

                for i,degree_array in enumerate(atom_neighbors_list):
                    for j,value in enumerate(degree_array):
                        atom_neighbors[atom_neighbor_count,j] = value
                    atom_neighbor_count += 1

            if len(bond_neighbors_list) > 0:
                for i,degree_array in enumerate(bond_neighbors_list):
                    for j,value in enumerate(degree_array):
                        bond_neighbors[bond_neighbor_count,j] = value
                    bond_neighbor_count += 1

        #then add everything to my arrays
        smiles_to_atom_info[smiles] = atoms
        smiles_to_bond_info[smiles] = bonds
        smiles_to_atom_index[smiles] = arrayrep['atom_index']
        smiles_to_atom_neighbors[smiles] = atom_neighbors
        smiles_to_bond_neighbors[smiles] = bond_neighbors
        
        smiles_to_atom_mask[smiles] = mask

    del smiles_to_fingerprint_features
    feature_dicts = {
        'smiles_to_atom_mask': smiles_to_atom_mask,
        'smiles_to_atom_info': smiles_to_atom_info,
        'smiles_to_bond_info': smiles_to_bond_info,
        'smiles_to_atom_neighbors': smiles_to_atom_neighbors,
        'smiles_to_bond_neighbors': smiles_to_bond_neighbors,
        'smiles_to_rdkit_list': smiles_to_rdkit_list,
        'smiles_to_atom-index':smiles_to_atom_index,
        'max_atom': max_atom_len
    }
    pickle.dump(feature_dicts,open(filename+'.pickle',"wb"))
    print('feature dicts file saved as '+ filename+'.pickle')
    return feature_dicts
def node_y_from_atom(smiles,x_atom,max_atom):
    mol = Chem.MolFromSmiles(smiles)
    num=len(mol.GetAtoms())
    lack_num_nodes = max_atom.shape[0] - num
    x_atom= torch.unsqueeze(torch.tensor([x for x in x_atom]), 1)
    empty_labels = torch.ones(lack_num_nodes, 1) * (-1)
    node_y = torch.cat((x_atom, empty_labels), 0)
    return node_y
def get_smiles_array(smilesList, feature_dicts):
    x_mask = []
    x_atom = []
    x_bonds = []
    x_atom_index = []
    x_bond_index = []
    adj=[]
    node_y=[]
    for smiles in smilesList:
        x_mask.append(feature_dicts['smiles_to_atom_mask'][smiles])
        x_atom.append(feature_dicts['smiles_to_atom_info'][smiles])
        x_bonds.append(feature_dicts['smiles_to_bond_info'][smiles])
        x_atom_index.append(feature_dicts['smiles_to_atom_neighbors'][smiles])
        x_bond_index.append(feature_dicts['smiles_to_bond_neighbors'][smiles])
        adj.append(bond_index_from_smiles(smiles,feature_dicts['smiles_to_bond_neighbors'][smiles]))
        node_y.append(node_y_from_atom(smiles,feature_dicts['smiles_to_atom-index'][smiles],feature_dicts['smiles_to_bond_neighbors'][smiles]))
    adj = torch.stack(adj, dim=0)
    node_y = torch.stack(node_y, dim=0)
    return np.asarray(x_atom),np.asarray(x_bonds),np.asarray(x_atom_index),\
        np.asarray(x_bond_index),np.asarray(x_mask),feature_dicts['smiles_to_rdkit_list'], adj ,node_y





