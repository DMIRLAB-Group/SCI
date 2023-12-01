import torch
import torch.nn as nn
from torch_sparse import SparseTensor
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
from GOOD.networks.models.GINs import GINMolEncoder, GINFeatExtractor, GINEConv
from GOOD.networks.models.GINvirtualnode import VirtualNodeEncoder, vGINEncoder
import torch_geometric.nn as gnn
from typing import Callable, Optional
from GOOD.networks.models.MolEncoders import AtomEncoder, BondEncoder
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor
from torch_geometric.nn.inits import reset
from GOOD.networks.models.BaseGNN import GNNBasic
from GOOD.networks.models.Classifiers import Classifier
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

@register.model_register
class SCI1(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(SCI1, self).__init__(config)
        self.dropout:float=config.model.dropout
        self.emb_dim1:int=config.model.emb_dim1
        self.emb_dim2:int=config.model.emb_dim2

        self.x_dim =9
        self.y_dim = 119
        self.S_emb_dim = config.model.dim_hidden
        self.emb_dim=config.model.dim_hidden
        self.dim_hidden=config.model.dim_hidden
        self.device=config.device
        self.gnn = vGINFeatExtractor(config)
        self.atom_encoder = AtomEncoder(self.emb_dim)
        self.z1_gcn = nn.ModuleList([DenseGCNConv(in_channels=self.emb_dim, out_channels=self.emb_dim1),
                                     DenseGCNConv(in_channels=self.emb_dim1, out_channels=self.emb_dim1)])
        self.z2_dc = BatchInnerProductDecoder()
        self.z5 = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=self.dropout),
                                    nn.Linear(self.emb_dim, self.emb_dim),
                                    nn.LeakyReLU())
        self.k1_MLP = nn.Linear(self.S_emb_dim, self.y_dim)
        self.g1_MLP = nn.Sequential(nn.Linear(2, 1),
                                    nn.Sigmoid())
        self.g2_MLP = nn.Sequential(nn.Linear(2, 1),
                                    nn.Sigmoid())
        self.X1_gcn = nn.ModuleList([DenseGCNConv(in_channels=self.S_emb_dim, out_channels=self.emb_dim2),
                                     DenseGCNConv(in_channels=self.emb_dim2, out_channels=self.emb_dim2)])
        self.X1_predict = nn.Linear(self.emb_dim2, self.x_dim)
        self.mu = GINEConv(nn.Sequential(nn.Linear(self.dim_hidden, 2 * self.dim_hidden),
                                            nn.BatchNorm1d(2 * self.dim_hidden), nn.ReLU(),
                                                nn.Dropout(p=self.dropout),
                                            nn.Linear(2 * self.dim_hidden, self.S_emb_dim)))
        self.logvar = GINEConv(nn.Sequential(nn.Linear(self.dim_hidden, 2 * self.dim_hidden),
                                            nn.BatchNorm1d(2 * self.dim_hidden), nn.ReLU(),
                                                    nn.Dropout(p=self.dropout),
                                            nn.Linear(2 * self.dim_hidden, self.S_emb_dim)))
        self.A_new_MLP = nn.Sequential(nn.Linear(2, 1),
                                       nn.Sigmoid())
        self.classifier = Classifier(config)

    def mlp_encode(self, x_emb, edge_index, edge_attr):
        emb = self.z5(x_emb)
        return self.mu(emb,edge_index,edge_attr), self.logvar(emb,edge_index,edge_attr)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def gcn_operate(self, z_gcn_module, x, adj, mask):
        z = F.relu(z_gcn_module[0](x, adj, mask=mask))
        z = F.dropout(z, self.dropout, training=self.training)
        z = F.relu(z_gcn_module[1](z, adj, mask=mask))
        return z

    def sampling(self, att_log_logits, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    def forward(self, *args, **kwargs):
        data = kwargs.get('data')
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x_emb = self.atom_encoder(x)

        mu, logvar = self.mlp_encode(x_emb, edge_index, edge_attr)
        s = self.reparameterize(mu, logvar).float()

        unique_elements, counts = torch.unique(batch, return_counts=True)
        num = torch.sum(torch.square(counts))

        adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(data.x.size(0), data.x.size(0)))
        adj = adj.to_dense()

        A = torch.sigmoid(torch.matmul(x_emb, x_emb.t())) * adj

        g_split = torch.split(x_emb, int(x_emb.size(1) / 2), dim=1)
        zh = g_split[0]
        gh_kl, gh = subgraph(zh, self.config.device, adj)
        gh_edge_index, gh_edge_attr = g_edge(gh, adj, data.edge_index, data.edge_attr)
        zn = g_split[1]
        gn_kl, gn = subgraph(zn, self.config.device, adj)

        node_y = data.x[:, 0].reshape(-1, 1)
        k1 = self.k1_MLP(s)

        g1 = torch.cat((gh.unsqueeze(dim=-1), gn.unsqueeze(dim=-1)), dim=-1)
        g1 = self.g1_MLP(g1)
        g1 = torch.squeeze(g1, dim=-1)*adj

        g2 = torch.cat((gn.unsqueeze(dim=-1), gh.unsqueeze(dim=-1)), dim=-1)
        g2 = self.g2_MLP(g2)
        g2 = torch.squeeze(g2, dim=-1)
        g2 = torch.where(g2 >= 0.5, torch.ones(1).to(self.device), torch.zeros(1).to(self.device))*adj
        x1 = self.X1_predict(self.gcn_operate(self.X1_gcn, s, g2, mask=None))

        z1 = self.gcn_operate(self.z1_gcn, x_emb, adj, mask=None)
        z2 = self.z2_dc(z1)
        z2 = z2.mul(adj)*adj

        indices = np.arange(gn.size(0))
        np.random.shuffle(indices)
        gn_new = gn[indices]
        A_new = torch.cat((gn_new.unsqueeze(dim=-1), gh.unsqueeze(dim=-1)), dim=-1)
        A_new = A_new.detach()
        A_new = self.A_new_MLP(A_new)
        A_new = torch.squeeze(A_new, dim=-1)
        A_new = torch.where(A_new >= 0.5, torch.ones(1).to(self.device), torch.zeros(1).to(self.device))*adj
        z1_A = self.gcn_operate(self.z1_gcn, x_emb, A_new, mask=None)
        z2_A = self.z2_dc(z1_A)
        z2_A = z2_A.mul(A_new)*adj

        gh_data = Munch()
        gh_data.x = s
        gh_data.edge_index = gh_edge_index
        gh_data.edge_attr = gh_edge_attr
        gh_data.batch = data.batch
        y = self.classifier(self.gnn(data=gh_data))

        return y,num,adj,A,gh,gn,node_y,k1,g1,x1,z2,z2_A,mu,logvar,gh_kl,gn_kl


class BatchInnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid):
        super(BatchInnerProductDecoder, self).__init__()
        self.act = act
    def forward(self, z):
        adj = self.act(torch.bmm(z, z.transpose(1, 2)))
        return adj

def sample_gumbel(device, shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(device, probs, temperature, eps=1e-5):
    y = torch.log(probs + eps) + sample_gumbel(device, probs.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(device, probs, temperature, latent_dim, categorical_dim=2):
    y = gumbel_softmax_sample(device, probs, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y.view(-1, latent_dim * categorical_dim), y_hard.view(-1, latent_dim * categorical_dim)

def subgraph(z, device, mask):
    g = torch.sigmoid(torch.matmul(z, z.t()))
    g = g.unsqueeze(0)
    g_one_probs = g.unsqueeze(3)
    g_zero_probs = 1 - g_one_probs
    g_node_num = g.size(1)
    z_new = torch.cat((g_zero_probs, g_one_probs), 3)

    g_kl, g = gumbel_softmax(device, probs=z_new, temperature=0.05, latent_dim=g_node_num * g_node_num,
                               categorical_dim=2)

    g_kl = torch.reshape(g_kl, (-1, g_node_num, g_node_num, 2))
    g = torch.reshape(g, (-1, g_node_num, g_node_num, 2))[:, :, :, 1]
    g = torch.reshape(g, (g_node_num, g_node_num)) * mask
    return g_kl,g

def adjacency_matrix(edge_index,device,num):
    v = torch.ones(edge_index.size(1)).float()
    adj_sparse = torch.sparse.FloatTensor(edge_index, v.to(device),
                                          torch.Size([num, num]))
    adj = adj_sparse.to_dense()
    return adj

def g_edge(g,adj,edge_index,edge_attr):
    g = g * adj
    g_index = torch.nonzero(g, as_tuple=False)
    edge = torch.cat((edge_index.t(), edge_attr), dim=1)
    g_edge = edge[torch.all(edge[:, :2].unsqueeze(1) == g_index.unsqueeze(0), dim=2).any(dim=1)]
    g_split = torch.split(g_edge, [edge_index.size(0), edge_attr.size(1)], dim=1)
    g_edge_index = g_split[0].t().to(torch.long)
    g_edge_attr = g_split[1].to(torch.long)
    return g_edge_index, g_edge_attr

class vGINFeatExtractor(GINFeatExtractor):
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(vGINFeatExtractor, self).__init__(config)
        num_layer = config.model.model_layer
        if config.dataset.dataset_type == 'mol':
            self.encoder = vGINMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = vGINEncoder(config, **kwargs)
            self.edge_feat = False

class vGINMolEncoder(GINMolEncoder, VirtualNodeEncoder):
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(vGINMolEncoder, self).__init__(config, **kwargs)
        self.config: Union[CommonArgs, Munch] = config

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch_size, device=self.config.device, dtype=torch.long))
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_attr))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch, batch_size) + virtual_node_feat)
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout
