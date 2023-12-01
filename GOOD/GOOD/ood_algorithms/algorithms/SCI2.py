from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
from .BaseOOD import BaseOODAlg
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F
from torch.distributions import kl_divergence

SMALL = 1e-5

@register.ood_alg_register
class SCI2(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(SCI2, self).__init__(config)
        self.num = None
        self.y_hat = None
        self.adj = None
        self.A = None
        self.g1 = None
        self.z2 = None
        self.z2_A = None
        self.node_y=None
        self.k1 = None
        self.x1 = None
        self.mu=None
        self.logvar=None
        self.gh_kl=None
        self.gn_kl=None
        self.gh=None
        self.gn=None
        self.y_loss_para = config.ood.y_loss_para
        self.g1_loss_para = config.ood.g1_loss_para
        self.k1_loss_para = config.ood.k1_loss_para
        self.x1_loss_para = config.ood.x1_loss_para
        self.gh_sparity_para=config.ood.gh_sparity_para
        self.KL_para=config.ood.KL_para
        self.Lr_para=config.ood.Lr_para
        self.gh_sparity_loss_para=config.ood.gh_sparity_loss_para
        self.gn_sparity_loss_para=config.ood.gn_sparity_loss_para


    def stage_control(self, config: Union[CommonArgs, Munch]):
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        self.y_hat,self.num,self.adj,self.A,self.gh,self.gn,\
            self.node_y,self.k1,self.g1,self.x1,self.z2, self.z2_A,\
            self.mu,self.logvar,self.gh_kl,self.gn_kl= model_output
        #
        #
        return self.y_hat

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        self.mean_loss = loss.mean()
        # y_hat = F.softmax(self.y_hat, dim=0)
        # y=(y_hat >= 0.5).float()

        # y_entropy = ce(y_hat, y)

        A_loss = torch.norm(self.A.float() - self.adj.float(), p=1) / self.num

        gh_sparity_loss = torch.norm(self.gh.float(), p=1) / self.num
        gn_sparity_loss = torch.norm(self.gn.float(), p=1) / self.num

        ce=nn.CrossEntropyLoss()
        node_y = self.node_y.squeeze(-1)
        k1_loss = ce(self.k1.float(), node_y.long())

        g1_loss = torch.norm(self.g1.float() - self.adj.float(), p=1) / self.num

        mse = nn.MSELoss()
        x1_loss = mse(self.x1.float().squeeze(), data.x.float())

        gh_A_loss = torch.square(torch.norm(self.z2.float() - self.z2_A.float(), p=2)) / self.num
        z2 = self.z2.double()
        z2_new = torch.where(z2 == 0, SMALL, z2)
        entropy = -torch.sum(z2_new * torch.log(z2_new)) / self.num
        Lr = gh_A_loss - entropy

        gh_sparity = self.gh_sparity_para * self.adj.sum() / self.num
        gn_sparity = (1 - self.gh_sparity_para) * self.adj.sum() / self.num
        kld_gh = self.calculate_kld(gh_sparity, self.gh_kl)
        kld_gn = self.calculate_kld(gn_sparity, self.gn_kl)
        kld = torch.mean(-0.5 * torch.sum(1 + self.logvar - self.mu ** 2 - self.logvar.exp(), dim=-1))

        loss = self.y_loss_para*self.mean_loss+A_loss+self.gh_sparity_loss_para*gh_sparity_loss+\
               self.gn_sparity_loss_para*gn_sparity_loss+self.k1_loss_para*k1_loss+self.g1_loss_para*g1_loss+\
               self.x1_loss_para * x1_loss+self.Lr_para*Lr+self.KL_para*(kld_gh+kld_gn+kld)
               #
               # +y_entropy
        return loss


    def calculate_kld(self,sparity, all_lag_structures: torch.tensor):
        posterior_dist = torch.distributions.Categorical(probs=all_lag_structures)
        adj_probs = torch.ones_like(all_lag_structures) * sparity
        adj_probs[:, :, :, 0] = 1 - adj_probs[:, :, :, 1]
        prior_dist = torch.distributions.Categorical(probs=adj_probs)
        KLD = kl_divergence(posterior_dist, prior_dist).mean()
        return KLD

    def set_up(self, model: torch.nn.Module, config: Union[CommonArgs, Munch]):
        self.model: torch.nn.Module = model
        self.optimizer = torch.optim.Adam([
            {'params': self.model.gnn.parameters(), 'lr': config.train.lr},
            {'params': self.model.classifier.parameters(), 'lr': config.train.lr},
            {'params': self.model.atom_encoder.parameters(), 'lr': config.train.lr},
            {'params': self.model.mu.parameters(), 'lr': config.train.decoder_lr},
            {'params': self.model.logvar.parameters(), 'lr': config.train.decoder_lr},
            {'params': self.model.z1_gcn.parameters(), 'lr': config.train.decoder_lr},
            {'params': self.model.z2_dc.parameters(), 'lr': config.train.decoder_lr},
            {'params': self.model.z5.parameters(), 'lr': config.train.decoder_lr},
            {'params': self.model.k1_MLP.parameters(), 'lr': config.train.decoder_lr},
            {'params': self.model.g1_MLP.parameters(), 'lr': config.train.decoder_lr},
            {'params': self.model.g2_MLP.parameters(), 'lr': config.train.decoder_lr},
            {'params': self.model.X1_gcn.parameters(), 'lr': config.train.decoder_lr},
            {'params': self.model.X1_predict.parameters(), 'lr': config.train.decoder_lr},
            {'params': self.model.A_new_MLP.parameters(), 'lr': config.train.decoder_lr}
        ], lr=config.train.lr, weight_decay=config.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.train.mile_stones,
                                                              gamma=0.1)
