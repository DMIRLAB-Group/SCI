import csv
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence


def calculate_kld(sparity, all_lag_structures: torch.tensor):
    posterior_dist = torch.distributions.Categorical(probs=all_lag_structures)
    adj_probs = torch.ones_like(all_lag_structures) * sparity
    adj_probs[:, :, :, 0] = 1 - adj_probs[:, :, :, 1]
    prior_dist = torch.distributions.Categorical(probs=adj_probs)

    KLD = kl_divergence(posterior_dist, prior_dist).mean()

    return KLD


def sample_gumbel(device, shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(device, probs, temperature, eps=1e-20):
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


def save_results_to_csv(headers, results, dataset):
    path = "graphClassification_" + dataset + "_results.csv"
    file_exist = True
    if not os.path.isfile(path):
        file_exist = False
    with open(path, 'a+') as f:
        csv_writer = csv.writer(f)
        if not file_exist:
            csv_writer.writerow(headers)
        csv_writer.writerow(results)


def feature_normalize(data):
    min_value = data.min(0)
    max_value = data.max(0)
    range_value = max_value - min_value
    col_index = np.array(np.nonzero(range_value)).flatten()

    data = data[:, col_index]
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mu) / std


def join_dataset(dataset0, dataset1):
    dataset0.x = np.concatenate((dataset0.x, dataset1.x), axis=0)
    dataset0.edge_attr = np.concatenate((dataset0.edge_attr, dataset1.edge_attr), axis=0)
    dataset0.edge_num = np.concatenate((dataset0.edge_num, dataset1.edge_num), axis=0)
    dataset0.node_y = np.concatenate((dataset0.node_y, dataset1.node_y), axis=0)
    dataset0.adj = np.concatenate((dataset0.adj, dataset1.adj), axis=0)
    dataset0.y = np.concatenate((dataset0.y, dataset1.y), axis=0)
    dataset0.num_nodes = np.concatenate((dataset0.num_nodes, dataset1.num_nodes), axis=0)
    return dataset0
