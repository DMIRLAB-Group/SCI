import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import random
import argparse
from loss import Loss_function
from spiltters import scaffold_split
import time
import numpy as np
import pickle
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.nn.Module.dump_patches = True
import pandas as pd
#then import my own modules
from SCI import SCI_model_plus, save_smiles_dicts, get_smiles_array
from sklearn.metrics import roc_auc_score
from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, help='which gpu to use if any (default: 0)')
parser.add_argument('--path', type=str, default='data/muv.csv', help='which path ')
parser.add_argument('--save_path', type=str, default='saved_models/', help='which save_path')
parser.add_argument('--epochs', type=int, help='number of epochs to train (default: 100)')
parser.add_argument('--batch_size', type=int, help='input batch size for training (default: 32)')
parser.add_argument('--p_dropout', type=float, help='dropout radio')
parser.add_argument('--fingerprint_dim', type=int, help='dimensionality of hidden units in encoder')
parser.add_argument('--emb_dim1', type=int,
                    help='input dimensionality of hidden units in mu_MLP and logvar_MLP')
parser.add_argument('--emb_dim2', type=int,
                    help='input dimensionality of hidden units in mu_MLP and logvar_MLP')
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

task_name = 'muv'
tasks = [
    "MUV-466","MUV-548","MUV-600","MUV-644","MUV-652","MUV-689","MUV-692","MUV-712","MUV-713","MUV-733","MUV-737","MUV-810","MUV-832","MUV-846","MUV-852","MUV-858","MUV-859"
]
raw_filename = args.path
feature_filename = raw_filename.replace('.csv','.pickle')
filename = raw_filename.replace('.csv','')
prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
smiles_tasks_df = pd.read_csv(raw_filename)
smilesList = smiles_tasks_df.smiles.values
print("number of all smiles: ",len(smilesList))
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
smiles_tasks_df['cano_smiles'] =canonical_smiles_list
assert canonical_smiles_list[8]==Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['cano_smiles'][8]), isomericSmiles=True)

start_time = str(time.ctime()).replace(':','-').replace(' ','_')
start = time.time()
np.random.seed(args.random_seed)
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.backends.cudnn.deterministic = True
per_task_output_units_num = 2 # for classification model with 2 classes
output_units_num = len(tasks) * per_task_output_units_num
smilesList = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms())<151]
if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb" ))
else:
    feature_dicts = save_smiles_dicts(smilesList,filename)

remained_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
uncovered_df = smiles_tasks_df.drop(remained_df.index)
weights = []
for i, task in enumerate(tasks):
    negative_df = remained_df[remained_df[task] == 0][["smiles", task]]
    positive_df = remained_df[remained_df[task] == 1][["smiles", task]]
    negative_train, negative_valid, negative_test, (negative_train_smiles, negative_valid_smiles, negative_test_smiles) = scaffold_split(negative_df, smilesList,
                                                                                            task_idx=None, null_value=0,
                                                                                            frac_train=0.8,
                                                                                            frac_valid=0.1,
                                                                                            frac_test=0.1,
                                                                                            return_smiles=True)

    positive_train, positive_valid, positive_test, (positive_train_smiles, positive_valid_smiles, positive_test_smiles) = scaffold_split(positive_df, smilesList,
                                                                                            task_idx=None, null_value=0,
                                                                                            frac_train=0.8,
                                                                                            frac_valid=0.1,
                                                                                            frac_test=0.1,
                                                                                            return_smiles=True)

    weights.append([(positive_test.shape[0] + negative_test.shape[0]) / negative_test.shape[0], \
                    (positive_test.shape[0] + negative_test.shape[0]) / positive_test.shape[0]])
    train_df_new = pd.concat([negative_train, positive_train])
    valid_df_new = pd.concat([negative_valid, positive_valid])
    test_df_new = pd.concat([negative_test, positive_test])
    if i == 0:
        train_df = train_df_new
        test_df = test_df_new
        valid_df = valid_df_new
    else:
        train_df = pd.merge(train_df, train_df_new, on='smiles', how='outer')
        test_df = pd.merge(test_df, test_df_new, on='smiles', how='outer')
        valid_df = pd.merge(valid_df, valid_df_new, on='smiles', how='outer')
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list, adj ,node_y = get_smiles_array([canonical_smiles_list[0]],
                                                                                             feature_dicts)
num_atom_features = x_atom.shape[-1]
num_bond_features = x_bonds.shape[-1]

loss_function = [nn.CrossEntropyLoss(torch.Tensor(weight), reduction='mean') for weight in weights]
model = SCI_model_plus(args.radius, args.T, num_atom_features, num_bond_features,
                    args.fingerprint_dim, output_units_num, args.p_dropout, args.emb_dim1, args.emb_dim2 , args.device)
model.cuda()
optimizer = optim.Adam(model.parameters(), 10 ** -args.learning_rate, weight_decay=10 ** -args.weight_decay)


def train(model, dataset, optimizer, loss_function):
    model.train()
    valList = np.arange(0, dataset.shape[0])
    # shuffle them
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], args.batch_size):
        batch = valList[i:i + args.batch_size]
        batch_list.append(batch)
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch, :]
        smiles_list = batch_df.smiles.values
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list, adj, node_y = get_smiles_array(
            smiles_list,
            feature_dicts)
        atoms_adj, g_A, gh, gn, z2, z2_A, x1, k1, mu, logvar, gh_kl, gn_kl, mol_prediction = model(torch.Tensor(x_atom),
                                                                                                   torch.Tensor(
                                                                                                       x_bonds),
                                                                                                   torch.cuda.LongTensor(
                                                                                                       x_atom_index),
                                                                                                   torch.cuda.LongTensor(
                                                                                                       x_bond_index),
                                                                                                   torch.Tensor(x_mask),
                                                                                                   torch.Tensor(adj),
                                                                                                   torch.Tensor(node_y))
        model.zero_grad()
        # Step 4. Compute your loss function. (Again, Torch wants the target wrapped in a variable)
        loss = 0.0
        gh_sparity = args.gh_sparity_para * adj.sum() / (adj.size(0) * adj.size(1) * adj.size(2))
        gn_sparity = (1 - args.gh_sparity_para) * adj.sum() / (adj.size(0) * adj.size(1) * adj.size(2))
        losstotal = Loss_function(adj, atoms_adj, g_A, gh, gn, z2, z2_A, x1, k1, x_atom, node_y, mu, logvar, gh_sparity,
                                  gn_sparity, gh_kl, gn_kl, args.KL_para, args.gn_sparity_loss_para,
                                  args.gh_sparity_loss_para, args.g1_loss_para, args.x1_loss_para, args.k1_loss_para,
                                  args.Lr_para, args.batch_size, args.device)
        for i, task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                                                     per_task_output_units_num]
            y_val = batch_df[task].values
            validInds = np.where((y_val == 0) | (y_val == 1))[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)
            loss += loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))
        # Step 5. Do the backward pass and update the gradient
        loss = loss*args.yg_loss_para + losstotal
        loss.backward()
        optimizer.step()

def eval(model, dataset):
    model.eval()
    y_val_list = {}
    y_pred_list = {}
    losses_list = []
    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], args.batch_size):
        batch = valList[i:i + args.batch_size]
        batch_list.append(batch)
    for counter, eval_batch in enumerate(batch_list):
        batch_df = dataset.loc[eval_batch, :]
        smiles_list = batch_df.smiles.values
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list, adj, node_y = get_smiles_array(
            smiles_list,
            feature_dicts)
        atoms_prediction, g_A, gh, gn, z2, z2_A, x1, k1, _, _, gh_kl, gn_kl, mol_prediction = model(
            torch.Tensor(x_atom), torch.Tensor(x_bonds),
            torch.cuda.LongTensor(x_atom_index),
            torch.cuda.LongTensor(x_bond_index), torch.Tensor(x_mask), torch.Tensor(adj), torch.Tensor(node_y))
        for i, task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                                                     per_task_output_units_num]
            y_val = batch_df[task].values
            validInds = np.where((y_val == 0) | (y_val == 1))[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)
            loss = loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))
            y_pred_adjust = F.softmax(y_pred_adjust, dim=-1).data.cpu().numpy()[:, 1]
            losses_list.append(loss.cpu().detach().numpy())
            try:
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
            except:
                y_val_list[i] = []
                y_pred_list[i] = []
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
    eval_roc = [roc_auc_score(y_val_list[i], y_pred_list[i]) for i in range(len(tasks))]
    eval_loss = np.array(losses_list).mean()

    return eval_roc, eval_loss

best_param = {}
best_param["roc_epoch"] = 0
best_param["loss_epoch"] = 0
best_param["valid_roc"] = 0
best_param["valid_loss"] = 9e8
best_param["test_roc"] = 0
best_param["test_loss"] = 9e8

for epoch in tqdm(range(args.epochs)):
    valid_roc, valid_loss = eval(model, valid_df)
    valid_roc_mean = np.array(valid_roc).mean()

    if valid_roc_mean > best_param["valid_roc"]:
        best_param["roc_epoch"] = epoch
        best_param["valid_roc"] = valid_roc_mean
        torch.save(model, args.save_path + prefix_filename + '_' + start_time + '_' + str(epoch) + '.pt')
    if valid_loss < best_param["valid_loss"]:
        best_param["loss_epoch"] = epoch
        best_param["valid_loss"] = valid_loss
    print("EPOCH:\t" + str(epoch) + '\n' \
          + "valid_roc" + ":" + str(valid_roc) + '\n' \
          + "valid_roc_mean" + ":" + str(valid_roc_mean) + '\n' \
          )
    if (epoch - best_param["roc_epoch"] > 6) and (epoch - best_param["loss_epoch"] > 8):
       break
    train(model, train_df, optimizer, loss_function)
# evaluate model
best_model = torch.load(
    args.save_path + prefix_filename + '_' + start_time + '_' + str(best_param["roc_epoch"]) + '.pt')
test_roc, test_losses = eval(best_model, test_df)
print("best epoch:"+str(best_param["roc_epoch"])
      +"\n"+"test_roc:"+str(test_roc)
      +"\n"+"test_roc_mean:",str(np.array(test_roc).mean())
     )
