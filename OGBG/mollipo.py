import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.nn.Module.dump_patches = True
from sklearn.metrics import mean_squared_error
import os
import torch
import torch.nn as nn
import argparse
import random
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from loss import Loss_function
import pandas as pd
from SCI import SCI_model, save_smiles_dicts, get_smiles_array
from sklearn.metrics import roc_auc_score
from rdkit import Chem
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', help='which dataset ')
parser.add_argument('--path', type=str, default='data/ogbg_mollipo', help='which path ')
parser.add_argument('--save_path', type=str, default='saved_models/model_', help='which save_path ')
parser.add_argument('--device', type=int, help='which gpu to use if any (default: 0)')
parser.add_argument('--epochs', type=int, help='number of epochs to train (default: 100)')
parser.add_argument('--batch_size', type=int, help='input batch size for training (default: 32)')
parser.add_argument('--p_dropout', type=float, help='dropout radio')
parser.add_argument('--fingerprint_dim', type=int, help='dimensionality of hidden units in encoder')
parser.add_argument('--emb_dim1', type=int, help='input dimensionality of hidden units in mu_MLP and logvar_MLP')
parser.add_argument('--emb_dim2', type=int, help='input dimensionality of hidden units in mu_MLP and logvar_MLP')
parser.add_argument('--radius', type=int, help='dimensionality of hidden units in encoder')
parser.add_argument('--T', type=int, help='dimensionality of hidden units in encoder')
parser.add_argument('--per_task_output_units_num', type=int, help='dimensionality of hidden units in encoder')
parser.add_argument('--learning_rate', type=float, help='learning rate')
parser.add_argument('--weight_decay', type=float, help='weight_decay')
parser.add_argument('--gh_sparity_para', type=float, help='parameter of gh_sparity')
parser.add_argument('--g1_loss_para', type=float, help='parameter of g1_loss')
parser.add_argument('--gh_sparity_loss_para', type=float, help='parameter of gh_sparity_loss')
parser.add_argument('--gn_sparity_loss_para', type=float, help='parameter of gn_sparity_loss')
parser.add_argument('--Lr_para', type=float, help='parameter of Lr')
parser.add_argument('--x1_loss_para', type=float, help='parameter of x1_loss')
parser.add_argument('--KL_para', type=float, help='parameter of KL')
parser.add_argument('--k1_loss_para', type=float, help='parameter of k1_loss')
parser.add_argument('--yg_loss_para', type=float, help='parameter of yg_loss')
parser.add_argument('--random_seed', type=int, help='the seed')

args = parser.parse_args()
task_name = args.dataset_name
raw_filename = f'{args.path}/mapping/mol.csv'
datasets = pd.read_csv(raw_filename, header=None, na_values=[''])
feature_filename = raw_filename.replace('.csv', '.pickle')
filename = raw_filename.replace('.csv', '')
prefix_filename = raw_filename.split('/')[-1].replace('.csv', '')
smiles_tasks_df = pd.read_csv(raw_filename)
smilesList = smiles_tasks_df.smiles.values
print("number of all smiles: ", len(smilesList))

if args.dataset_name == 'lipo':
    tasks = ['exp']
    args.per_task_output_units_num = 1
atom_num_dist = []
remained_smiles = []
canonical_smiles_list = []
for smiles in smilesList:
    try:
        mol = Chem.MolFromSmiles(smiles)
        atom_num_dist.append(len(mol.GetAtoms()))
        remained_smiles.append(smiles)
        canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
    except:
        print("not successfully processed smiles: ", smiles)
        pass

print("number of successfully processed smiles: ", len(remained_smiles))
smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
smiles_tasks_df['cano_smiles'] = canonical_smiles_list
assert canonical_smiles_list[8] == Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['cano_smiles'][8]),
                                                    isomericSmiles=True)

def train(model, dataset, optimizer, loss_function):
    model.train()
    valList = np.arange(0, dataset.shape[0])
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], args.batch_size):
        batch = valList[i:i + args.batch_size]
        batch_list.append(batch)
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch, :]
        smiles_list = batch_df.cano_smiles.values
        x_atom, x_bonds, x_atom_index, x_bond_index, \
        x_mask, smiles_to_rdkit_list, adj, node_y = get_smiles_array(smiles_list,feature_dicts)
        atoms_adj, g_A, gh, gn, z2, z2_A, x1, k1, mu, logvar, \
        gh_kl, gn_kl, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),
                                             torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),
                                            torch.Tensor(x_mask),torch.Tensor(adj),torch.Tensor(node_y))
        model.zero_grad()
        loss = 0.0
        gh_sparity = args.gh_sparity_para * adj.sum() / (adj.size(0) * adj.size(1) * adj.size(2))
        gn_sparity = (1 - args.gh_sparity_para) * adj.sum() / (adj.size(0) * adj.size(1) * adj.size(2))
        losstotal = Loss_function(adj, atoms_adj, g_A, gh, gn, z2, z2_A, x1, k1, x_atom, node_y, mu, logvar, gh_sparity,
                                  gn_sparity, gh_kl, gn_kl, args.KL_para, args.gn_sparity_loss_para,
                                  args.gh_sparity_loss_para, args.g1_loss_para, args.x1_loss_para, args.k1_loss_para,
                                  args.Lr_para, args.batch_size, args.device)
        for i, task in enumerate(tasks):
            y_pred = mol_prediction[:, i * args.per_task_output_units_num:(i + 1) *
                                                                          args.per_task_output_units_num]
            y_val = batch_df[task].values
            if args.per_task_output_units_num == 2:
                validInds = np.where((y_val == 0) | (y_val == 1))[0]
                if len(validInds) == 0:
                    continue
                y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
                validInds = torch.cuda.LongTensor(validInds).squeeze()
                y_pred_adjust = torch.index_select(y_pred, 0, validInds)
                loss += loss_function[i](
                    y_pred_adjust,
                    torch.cuda.LongTensor(y_val_adjust))
            elif args.per_task_output_units_num == 1:
                y_pred_adjust = y_pred
                y_val_adjust = y_val
                loss += torch.sqrt(loss_function[i](
                    y_pred_adjust,
                    torch.Tensor(y_val_adjust).unsqueeze(1).to(args.device)))
        # Step 5. Do the backward pass and update the gradient
        losstotal = loss * args.yg_loss_para + losstotal
        losstotal.backward()
        optimizer.step()

def eval(model, dataset):
    model.eval()
    y_val_list = []
    y_pred_list = []

    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], args.batch_size):
        batch = valList[i:i + args.batch_size]
        batch_list.append(batch)
    for counter, test_batch in enumerate(batch_list):
        batch_df = dataset.loc[test_batch, :]
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list, adj, node_y = get_smiles_array(
            smiles_list,
            feature_dicts)
        atoms_prediction, g_A, gh, gn, z2, z2_A, x1, k1, _, _, gh_kl, gn_kl, mol_prediction = model(
            torch.Tensor(x_atom), torch.Tensor(x_bonds),
            torch.cuda.LongTensor(x_atom_index),
            torch.cuda.LongTensor(x_bond_index), torch.Tensor(x_mask), torch.Tensor(adj), torch.Tensor(node_y))
        for i, task in enumerate(tasks):
            y_pred = mol_prediction[:, i * args.per_task_output_units_num:(i + 1) *
                                                                     args.per_task_output_units_num]
            y_val = batch_df[task].values

            if args.per_task_output_units_num == 2:
                validInds = np.where((y_val == 0) | (y_val == 1))[0]
                if len(validInds) == 0:
                    continue
                y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
                validInds = torch.LongTensor(validInds).to(args.device).squeeze()
                y_pred_adjust = torch.index_select(y_pred, 0, validInds)
                y_pred_adjust = F.softmax(y_pred_adjust, dim=-1).data.cpu().numpy()[:, 1]
            elif args.per_task_output_units_num == 1:
                mask, y_val = nan2zero_get_mask(torch.from_numpy(y_val))
                y_pred, y_val = eval_data_preprocess(y_val, y_pred, mask)
                y_val_adjust = torch.Tensor(y_val).unsqueeze(1).to(args.device)
                y_pred_adjust = y_pred

            y_val_list.append(torch.Tensor(y_val_adjust).cpu())
            y_pred_list.append(torch.Tensor(y_pred_adjust).cpu())

    if args.per_task_output_units_num == 2:
        test_score = eval_score(y_pred_list, y_val_list, roc_auc_score)
    elif args.per_task_output_units_num == 1:
        test_score = eval_score(y_pred_list, y_val_list, mean_squared_error)
    return test_score

def nan2zero_get_mask(y):

    mask = ~torch.isnan(y)
    if mask is None:
        return None, None
    targets = torch.clone(y).detach()
    assert mask.shape[0] == targets.shape[0]
    mask = mask.reshape(targets.shape)
    targets[~mask] = 0

    return mask, targets

def eval_data_preprocess(y, raw_pred, mask):

    if args.per_task_output_units_num == 2:
        pred_prob = raw_pred.sigmoid()
        if y.shape[1] > 1:
            preds = []
            targets = []
            for i in range(y.shape[1]):
                preds.append(pred_prob[:, i][mask[:, i]].detach().cpu().numpy())
                targets.append(y[:, i][mask[:, i]].detach().cpu().numpy())
            return preds, targets
        pred = pred_prob[mask].reshape(-1).detach().cpu().numpy()
    elif args.per_task_output_units_num == 1:
        pred = raw_pred[mask].reshape(-1).detach().cpu().numpy()
    else:
        raise ValueError('Dataset task value error.')

    target = y[mask].reshape(-1).detach().cpu().numpy()

    return pred, target

def eval_score(pred_all, target_all, score_func):

    np.seterr(invalid='ignore')
    assert type(pred_all) is list, 'Wrong prediction input.'
    if type(pred_all[0]) is list:
        # multi-task
        all_task_preds = []
        all_task_targets = []
        for task_i in range(len(pred_all[0])):
            preds = []
            targets = []
            for pred, target in zip(pred_all, target_all):
                preds.append(pred[task_i])
                targets.append(target[task_i])
            all_task_preds.append(np.concatenate(preds))
            all_task_targets.append(np.concatenate(targets))

        scores = []
        for i in range(len(all_task_preds)):
            if all_task_targets[i].shape[0] > 0:
                scores.append(np.nanmean(score_func(all_task_targets[i], all_task_preds[i])))
        score = np.nanmean(scores)
    else:
        pred_all = np.concatenate(pred_all)
        target_all = np.concatenate(target_all)
        score = np.nanmean(score_func(target_all, pred_all))
    return score

def load_data(args, process):
    raw_filename = f'{args.path}/{process}.csv'
    feature_filename = raw_filename.replace('.csv', '.pickle')
    filename = raw_filename.replace('.csv', '')
    smiles_tasks_df = pd.read_csv(raw_filename)
    smilesList = smiles_tasks_df.smiles.values
    print("number of all smiles: ", len(smilesList))
    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            mol = Chem.MolFromSmiles(smiles)
            atom_num_dist.append(len(mol.GetAtoms()))
            remained_smiles.append(smiles)
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            print("not successfully processed smiles: ", smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))
    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
    smiles_tasks_df['cano_smiles'] = canonical_smiles_list
    assert canonical_smiles_list[8] == Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['cano_smiles'][8]),
                                                        isomericSmiles=True)
    smilesList = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms()) < 101]
    uncovered = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms()) > 100]
    smiles_tasks_df = smiles_tasks_df[~smiles_tasks_df["cano_smiles"].isin(uncovered)]
    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb"))
    else:
        feature_dicts = save_smiles_dicts(smilesList, filename)
    remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    return remained_df

def reset_random_seed(seed):

    # Fix Random seed
    random.seed(seed)
    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    # Default state is a training state
    torch.enable_grad()

print("begin!")
reset_random_seed(args.random_seed)
output_units_num = len(tasks) * args.per_task_output_units_num
smilesList = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms()) < 101]
uncovered = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms()) > 100]
smiles_tasks_df = smiles_tasks_df[~smiles_tasks_df["cano_smiles"].isin(uncovered)]

if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb"))
else:
    feature_dicts = save_smiles_dicts(smilesList, filename)

remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
uncovered_df = smiles_tasks_df.drop(remained_df.index)

weights = []
for i, task in enumerate(tasks):
    if args.per_task_output_units_num == 2:
        negative_df = remained_df[remained_df[task] == 0][["smiles", task]]
        positive_df = remained_df[remained_df[task] == 1][["smiles", task]]
        weights.append([(positive_df.shape[0] + negative_df.shape[0]) / negative_df.shape[0], \
                        (positive_df.shape[0] + negative_df.shape[0]) / positive_df.shape[0]])
    elif args.per_task_output_units_num == 1:
        weights.append(i)

train_df = load_data(args, 'train').reset_index(drop=True)
valid_df = load_data(args, 'valid').reset_index(drop=True)
test_df = load_data(args, 'test').reset_index(drop=True)

x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list, adj ,node_y = get_smiles_array([canonical_smiles_list[0]],
                                                                                             feature_dicts)
num_atom_features = x_atom.shape[-1]
num_bond_features = x_bonds.shape[-1]
if args.per_task_output_units_num == 2:
    loss_function = [nn.CrossEntropyLoss(torch.Tensor(weight), reduction='mean') for weight in weights]
elif args.per_task_output_units_num == 1:
    loss_function = [nn.MSELoss(reduction="mean") for weight in weights]
model = SCI_model(args.radius, args.T, num_atom_features, num_bond_features,
                    args.fingerprint_dim, output_units_num, args.p_dropout, args.emb_dim1, args.emb_dim2 , args.device)
model.to(args.device)
optimizer = optim.Adam(model.parameters(), 10 ** -args.learning_rate, weight_decay=10 ** -args.weight_decay)

if args.per_task_output_units_num == 2:
    t = 1
elif args.per_task_output_units_num == 1:
    t = -1
for epoch in tqdm(range(args.epochs)):
    train(model, train_df, optimizer, loss_function)
    valid_mean = eval(model, valid_df)
    print(f'epoch:{epoch} valid_score:{valid_mean}')
    if epoch == 0:
        best_valid = valid_mean
    if valid_mean * t >= best_valid * t:
        best_valid = valid_mean
        test_mean = eval(model, test_df)
        torch.save(model, args.save_path + prefix_filename + '_' + str(args.random_seed) + '.pt')
        print(f'test_score:{test_mean}')
