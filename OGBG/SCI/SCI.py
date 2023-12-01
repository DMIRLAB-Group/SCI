import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import DenseGCNConv,DenseGINConv
import numpy as np
from utils import *

class SCI_model(nn.Module):

    def __init__(self, radius, T, input_feature_dim, input_bond_dim,\
            fingerprint_dim, output_units_num, p_dropout, emb_dim1, emb_dim2 ,device):
        super(SCI_model, self).__init__()
        self.emb_dim1 = emb_dim1
        self.emb_dim2 = emb_dim2
        self.device = device
        # graph attention for atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc = nn.Linear(input_feature_dim+input_bond_dim, fingerprint_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2*fingerprint_dim,1) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)])
        # graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2*fingerprint_dim,1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)
        # you may alternatively assign a different set of parameter in each attentive layer for molecule embedding like in atom embedding process.
        self.dropout = nn.Dropout(p=p_dropout)
        self.output = nn.Linear(fingerprint_dim, output_units_num)
        self.sigmoid = nn.Sigmoid()
        self.radius = radius
        self.T = T

        self.z1_gcn = nn.ModuleList([DenseGINConv(nn.Sequential(nn.Linear(fingerprint_dim, self.emb_dim1))),
                                     DenseGINConv(nn.Sequential(nn.Linear(self.emb_dim1, self.emb_dim1)))])
        self.bn1 = nn.BatchNorm1d(self.emb_dim1)
        self.relu1 = nn.ReLU()

        self.z2_dc = BatchInnerProductDecoder()
        self.z3_gcn = nn.ModuleList([DenseGINConv(nn.Sequential(nn.Linear(fingerprint_dim, self.emb_dim1))),
                                     DenseGINConv(nn.Sequential(nn.Linear(self.emb_dim1, self.emb_dim1)))])
        self.z4_dc = BatchInnerProductDecoder()
        self.X1_gcn = nn.ModuleList([DenseGINConv(nn.Sequential(nn.Linear(fingerprint_dim, self.emb_dim1))),
                                     DenseGINConv(nn.Sequential(nn.Linear(self.emb_dim1, self.emb_dim1)))])
        self.X1_predict = nn.Linear(self.emb_dim1, self.emb_dim2)
        self.g1_MLP = nn.Sequential(nn.Linear(2, 1),
                                    nn.Sigmoid())
        self.A_new_MLP = nn.Sequential(nn.Linear(2, 1),
                                       nn.Sigmoid())
        self.g2_MLP = nn.Sequential(nn.Linear(2, 1),
                                    nn.Sigmoid())
        self.z5_MLP = nn.Sequential(nn.Linear(fingerprint_dim, fingerprint_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p = (self.dropout.p)),
                                    nn.Linear(fingerprint_dim, fingerprint_dim),
                                    nn.LeakyReLU())
        self.s_linear = nn.Linear(emb_dim1, fingerprint_dim)
        self.mu_linear = nn.Linear(fingerprint_dim, emb_dim1)
        self.logvar_linear = nn.Linear(fingerprint_dim, emb_dim1)
        self.k1_MLP = nn.Sequential(nn.Linear(emb_dim1, fingerprint_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p = (self.dropout.p)),
                                    nn.Linear(fingerprint_dim, self.emb_dim1))
        self.s_gcn = nn.ModuleList([DenseGCNConv(in_channels=fingerprint_dim, out_channels=fingerprint_dim)])
    def bn_operator(self, bn, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.reshape(-1, num_channels)
        x = bn(x)
        x = x.reshape(batch_size, num_nodes, num_channels)
        return x
    def mlp_encode(self, fingerprint_dim):
        emb = self.z5_MLP(fingerprint_dim)
        return self.mu_linear(emb), self.logvar_linear(emb)
    def gcn_operate(self, z_gcn_module, x, adj, mask):
        z = F.relu(z_gcn_module[0](x, adj, mask=mask))
        z = self.bn_operator(self.bn1, z)
        z = F.dropout(z, self.dropout.p, training=self.training)
        z = F.relu(z_gcn_module[1](z, adj, mask=mask))
        return z
    def gcn1_operate(self, z_gcn_module, x, adj):
        z = F.relu(z_gcn_module[0](x, adj))
        z = F.dropout(z, self.dropout.p, training=self.training)
        return z
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, adj, node_y):
        atom_mask = atom_mask.unsqueeze(2)
        mask_adj = torch.where(node_y == -1, 0, 1).squeeze(-1)
        batch_size, mol_length, num_atom_feat = atom_list.size()
        atom_feature = F.leaky_relu(self.atom_fc(atom_list))

        # encoder
        z1 = self.gcn_operate(self.z1_gcn, atom_feature, adj, mask=mask_adj)
        z2 = self.z2_dc(z1)
        z2 = z2.mul(adj)
        Gh_one_probs = z2.unsqueeze(3)
        Gh_zero_probs = 1 - Gh_one_probs
        node_num = z2.size(1)
        z2_new = torch.cat((Gh_zero_probs, Gh_one_probs), 3)
        gh_kl, gh = gumbel_softmax(self.device, probs=z2_new, temperature=0.05, latent_dim=node_num * node_num,
                                   categorical_dim=2)
        gh = torch.reshape(gh, (-1, node_num, node_num, 2))[:, :, :, 1]
        gh_kl = torch.reshape(gh_kl, (-1, node_num, node_num, 2))

        z3 = self.gcn_operate(self.z3_gcn, atom_feature, adj, mask=mask_adj)
        z4 = self.z4_dc(z3)
        z4 = z4.mul(adj)
        Gn_one_probs = z4.unsqueeze(3)
        Gn_zero_probs = 1 - Gn_one_probs
        node_num = z4.size(1)
        z4_new = torch.cat((Gn_zero_probs, Gn_one_probs), 3)
        gn_kl, gn = gumbel_softmax(self.device, probs=z4_new, temperature=0.05, latent_dim=node_num * node_num,
                                   categorical_dim=2)
        gn = torch.reshape(gn, (-1, node_num, node_num, 2))[:, :, :, 1]
        gn_kl = torch.reshape(gn_kl, (-1, node_num, node_num, 2))

        # encoder
        g_A = torch.cat((gn.unsqueeze(dim=-1), gh.unsqueeze(dim=-1)), dim=-1)
        g_A = self.g1_MLP(g_A)
        g_A = torch.squeeze(g_A, dim=-1)

        # shuffle
        indices = np.arange(gn.size(0))
        np.random.shuffle(indices)
        gn_new = gn[indices]
        A_new = torch.cat((gn_new.unsqueeze(dim=-1), gh.unsqueeze(dim=-1)), dim=-1)
        A_new = A_new.detach()
        A_new = self.A_new_MLP(A_new)
        A_new = torch.squeeze(A_new, dim=-1)
        A_new = torch.where(A_new >= 0.5, torch.ones(1).to(self.device), torch.zeros(1).to(self.device))
        z1_A = self.gcn_operate(self.z1_gcn, atom_feature, A_new, mask=mask_adj)
        z2_A = self.z2_dc(z1_A)
        z2_A = z2_A.mul(A_new)

        # encoder
        mu, logvar = self.mlp_encode(atom_feature)
        s = self.reparameterize(mu, logvar)

        # decoder
        k1 = self.k1_MLP(s)

        # decoder
        s = self.s_linear(s)
        g2 = torch.cat((gn.unsqueeze(dim=-1), gh.unsqueeze(dim=-1)), dim=-1)
        g2 = self.g2_MLP(g2)
        g2 = torch.squeeze(g2, dim=-1)
        g2 = torch.where(g2 >= 0.5, torch.ones(1).to(self.device), torch.zeros(1).to(self.device))
        x1 = self.X1_predict(self.gcn_operate(self.X1_gcn, s, g2, mask=mask_adj))

        bond_neighbor = [bond_list[i][bond_degree_list[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [atom_list[i][atom_degree_list[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        # then concatenate them
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor],dim=-1)
        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor_feature))

        # generate mask to eliminate the influence of blank atoms
        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length-1] = 1
        attend_mask[attend_mask == mol_length-1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)
        softmax_mask = atom_degree_list.clone()

        softmax_mask[softmax_mask != mol_length-1] = 0
        softmax_mask[softmax_mask == mol_length-1] = -9e8 # make the softmax value extremly small
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)
        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score,-2)
        attention_weight = attention_weight * attend_mask
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
        context = F.elu(context)
        context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)

        #do nonlinearity
        activated_features = F.relu(s)

        for d in range(self.radius-1):
            neighbor_feature = [activated_features[i][atom_degree_list[i]] for i in range(batch_size)]
            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
            feature_align = torch.cat([atom_feature_expand, neighbor_feature],dim=-1)
            align_score = F.leaky_relu(self.align[d+1](self.dropout(feature_align)))
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score,-2)

            attention_weight = attention_weight * attend_mask
            neighbor_feature_transform = self.attend[d+1](self.dropout(neighbor_feature))
            context = torch.sum(torch.mul(attention_weight,neighbor_feature_transform),-2)
            context = F.elu(context)
            context_reshape = context.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d+1](context_reshape, atom_feature_reshape)
            atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)

            # do nonlinearity
            activated_features = F.relu(atom_feature)

        activated_features = self.gcn1_operate(self.s_gcn, activated_features, gh)
        mol_feature = torch.sum(activated_features * atom_mask, dim=-2)
        atom_feature_T = atom_feature.permute(0, 2, 1)
        atom_feature_A = torch.bmm(atom_feature, atom_feature_T)
        atom_feature_A = self.sigmoid (atom_feature_A)
        # do nonlinearity
        activated_features_mol = F.relu(mol_feature)
        mol_softmax_mask = atom_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)
        # feature extractor mol_feature -->decoder --> molecole

        for t in range(self.T):
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, mol_length, fingerprint_dim)
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = F.leaky_relu(self.mol_align(mol_align))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score,-2)
            mol_attention_weight = mol_attention_weight * atom_mask
            activated_features_transform = self.mol_attend(self.dropout(activated_features))
            mol_context = torch.sum(torch.mul(mol_attention_weight,activated_features_transform),-2)
            mol_context = F.elu(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)

            # do nonlinearity
            activated_features_mol = F.relu(mol_feature)

        mol_prediction = self.output(self.dropout(mol_feature))
        return atom_feature_A, g_A, gh, gn, z2, z2_A, x1 , k1, mu, logvar, gh_kl, gn_kl, mol_prediction

class BatchInnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid):
        super(BatchInnerProductDecoder, self).__init__()
        self.act = act

    def forward(self, z):
        adj = self.act(torch.bmm(z, z.transpose(1, 2)))
        return adj