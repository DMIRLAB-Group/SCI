import torch
import torch.nn as nn
import torch.nn.modules.loss
from utils import calculate_kld

SMALL = 1e-5


def Loss_function(adj, atoms_adj, g_A, gh, gn, z2, z2_A, x1, k1, x_atom, node_y, mu, logvar, gh_sparity, gn_sparity, gh_kl, gn_kl, kl_para, gn_sparity_loss_para, gh_sparity_loss_para, g1_loss_para, x1_loss_para, k1_loss_para, Lr_para, batch_size, device):

    g_A_loss = torch.norm(g_A.float() - adj.float(), p=1) / (batch_size * adj.shape[2] * adj.shape[2])

    adj_loss = torch.norm(adj.float() - atoms_adj.float(), p=1) / (batch_size * adj.shape[2] * adj.shape[2])

    shuffle_A_loss = torch.square(torch.norm(z2.float() - z2_A.float(), p=2)) / (
                batch_size * adj.shape[2] * adj.shape[2])

    z2 = z2.double()
    batch_size = z2.size(0)
    node_num = z2.size(1)
    z2_new = torch.where(z2 == 0, SMALL, z2)
    entropy = -torch.sum(z2_new * torch.log(z2_new)) / (batch_size * node_num * node_num)
    Lr = shuffle_A_loss - entropy

    mse1 = nn.MSELoss()
    node_y = node_y.squeeze(-1)
    is_node_label = node_y != -1
    ce1 = nn.CrossEntropyLoss()
    k1_loss = ce1(k1.float()[is_node_label], node_y.long()[is_node_label])
    x1_loss = mse1(x1.float().to(device)[is_node_label],
                   torch.from_numpy(x_atom).float().to(device)[is_node_label])
    kld_gh = calculate_kld(gh_sparity, gh_kl)
    kld_gn = calculate_kld(gn_sparity, gn_kl)
    kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1))
    gh_sparity_loss = torch.norm(gh.float(), p=1) / (batch_size * adj.shape[2] * adj.shape[2])
    gn_sparity_loss = torch.norm(gn.float(), p=1) / (batch_size * adj.shape[2] * adj.shape[2])
    losstotal = g1_loss_para*g_A_loss + adj_loss + x1_loss_para*x1_loss + kl_para * (kld_gh + kld_gn + kld) + gh_sparity_loss_para*gh_sparity_loss + Lr_para*Lr + gn_sparity_loss_para*gn_sparity_loss + k1_loss*k1_loss_para
    return losstotal
