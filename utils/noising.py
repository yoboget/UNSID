import os.path
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.mixture_same_family import MixtureSameFamily


class NoiserDirichlet:
    def __init__(self, timesteps, prior, distributions=None, mask_prior=False, scale=None):

        self.T = timesteps
        self.prior = prior
        self.scale = scale
        if distributions is not None:
            self.marginal_x = distributions[0]
            self.marginal_a = distributions[1]

        self.mask_prior = mask_prior

        self.base_params = torch.tensor([3.0, 1.0])

    def __call__(self, x, a, mask, t=None, t_node=None, t_edge=None):
        self.device = a.device

        if t is None:
            t = torch.rand((x.size(0),)).to(self.device)
        if t_node is None:
            t_node = t.to(self.device)
        if t_edge is None:
            t_edge = t.to(self.device)

        x = self.noise_x(x, t_node) * mask.unsqueeze(-1)
        diag = 1-torch.eye(a.shape[-2]).to(self.device).unsqueeze(0).unsqueeze(-1)
        a = self.noise_adj(a, t_edge) * mask[..., None, None] * mask[:, None, :, None] * diag

        return x, a, t.view(-1, 1)

    def noise_x(self, x, t):
        if x.size(-1) > 1:
            if self.prior == 'dirch':
                alphas = -(torch.log(torch.tensor(1.0, device=x.device) - t) * self.scale)
                alpha = alphas.view(-1, 1, 1) * torch.ones_like(x).to(self.device)
                ones = torch.ones(x.size(), device=x.device)
                alpha_i = alpha * x
                alpha_ = alpha_i + ones
                x = Dirichlet(alpha_).sample()

            elif self.prior == 'dirch_k':
                k = x.size(-1)
                # params = torch.ones(x.size(), device=x.device) * k * (1-t).view(-1, 1, 1) * (1-x)
                # data = x * k
                params = torch.ones(x.size(), device=x.device) * (1 - t).view(-1, 1, 1) * (1 - x)
                params += x
                # alpha_ = data + not_data
                x = Dirichlet(params).sample()

            elif self.prior == 'dirch_marg':
                k = x.size(-1)
                alpha_i = -(torch.log(torch.tensor(1.0, device=x.device) - t) * self.scale)
                # It K - 1 but one is added next lines
                alpha_i = (k - 2 + alpha_i.view(-1, 1, 1) * torch.ones_like(x).to(self.device)) * x
                ones = torch.ones(x.size(), device=x.device)
                params = alpha_i + ones
                x = Dirichlet(params).sample()

            elif self.prior == 'dirch_marg_t':
                k = x.size(-1)
                alpha_i = -(t * torch.log(torch.tensor(1.0, device=x.device) - t) * self.scale)
                # It is K - 1 but one is added next lines
                alpha_i = (k - 2 + alpha_i.view(-1, 1, 1) * torch.ones_like(x).to(self.device)) * x
                ones = torch.ones(x.size(), device=x.device)
                params = alpha_i + ones
                x = Dirichlet(params).sample()

            elif self.prior == 'dirch_mix':
                k = x.size(-1)
                assert k >= 2
                if k <= 3:
                    k = k
                mix_params = t * x + (1 - t) * self.marginal_x
                mix = Categorical(mix_params)
                alpha = -(torch.log(torch.tensor(1.0, device=x.device) - t) * self.scale)
                dirichlet_params = alpha * torch.eye(k, device=x.device).view(-1, 1, 1)
                dirichelet = Dirichlet(dirichlet_params)
                x = MixtureSameFamily(mix, dirichelet).sample()

            elif self.prior == 'normal':
                alphas = -(torch.log(torch.tensor(1.0, device=x.device) - t) * self.scale)
                x_0 = torch.randn(x.size(), device=x.device)
                x_1 = x * 2 - 1
                x = alphas.view(-1, 1, 1) * x_1 + (1 - alphas.view(-1, 1, 1)) * x_0

        t = t.view(-1, 1) * torch.ones(x.size(0), x.size(1)).to(self.device)
        x = torch.cat((x, t[..., None], 1-t[..., None]), dim=-1)
        return x

    def noise_adj(self, a, t):
        if self.prior == 'dirch':
            alphas = -(torch.log(torch.tensor(1.0, device=a.device) - t) * self.scale)
            alpha = alphas.view(-1, 1, 1, 1) * torch.ones_like(a).to(self.device)
            ones = torch.ones(a.size(), device=a.device)
            alpha_i = alpha * a
            alpha_ = alpha_i + ones
            a = Dirichlet(alpha_).sample()
        elif self.prior == 'dirch_k':
            params = torch.ones(a.size(), device=a.device) * (1 - t).view(-1, 1, 1, 1) * (1 - a)
            params += a
            # alpha_ = data + not_data
            a = Dirichlet(params).sample()
        elif self.prior == 'dirch_marg':
            k = a.size(-1)
            assert k >= 2
            if a.size(-1) <= 2:
                k = 3
            alpha_i = -(torch.log(torch.tensor(1.0, device=a.device) - t) * self.scale)
            # It K - 1 but one is added next lines
            alpha_i = (k - 2 + alpha_i.view(-1, 1, 1, 1) * torch.ones_like(a).to(self.device))
            alpha_i = alpha_i * a
            ones = torch.ones(a.size(), device=a.device)
            params = alpha_i + ones
            a = Dirichlet(params).sample()

        elif self.prior == 'dirch_marg_t':
            k = a.size(-1)
            assert k >= 2
            if k == 2:
                k = 3
            alpha_i = -(t * torch.log(torch.tensor(1.0, device=a.device) - t) * self.scale)
            # It K - 1 but one is added next lines
            alpha_i = (k - 2 + alpha_i.view(-1, 1, 1, 1) * torch.ones_like(a).to(self.device))
            alpha_i = alpha_i * a
            ones = torch.ones(a.size(), device=a.device)
            params = alpha_i + ones
            a = Dirichlet(params).sample()

        elif self.prior == 'dirch_mix':
            bs, n, _, d = a.shape
            mix_params = t.view(-1, 1, 1, 1) * a + (1 - t.view(-1, 1, 1, 1)) * self.marginal_a.to(a.device).view(1, 1,
                                                                                                                 1, d)
            # Normalize to get valid probabilities
            mix_params = mix_params / mix_params.sum(-1, keepdim=True)
            mix = Categorical(mix_params)
            alpha = -(torch.log(torch.tensor(1.0, device=a.device) - t) * self.scale)  # (bs,)
            dirichlet_params = 1 + alpha.view(-1, 1, 1, 1, 1) * torch.eye(d, device=a.device).view( 1, 1, 1, d, d)
            dirichlet_params = dirichlet_params.expand(bs, n, n, d, d)
            dirichlet = Dirichlet(dirichlet_params)
            a = MixtureSameFamily(mix, dirichlet).sample()

        elif self.prior == 'dirch2':
            alphas = -(torch.log(torch.tensor(1.0, device=a.device) - t) * self.scale)
            alpha = alphas.view(-1, 1, 1, 1) * torch.ones_like(a).to(self.device)
            ones = torch.ones(a.size(), device=a.device)
            alpha = alpha * a + ones
            alpha = t.view(-1, 1, 1, 1) * alpha + (1 - t.view(-1, 1, 1, 1)) * self.base_params.view(1, 1, 1, -1).to(self.device)
            a = Dirichlet(alpha).sample()

        elif self.prior == 'abs':
            ones = torch.ones(a.size(), device=a.device)

            alphas = -(torch.log(torch.tensor(1.0, device=a.device) - t) * self.scale)
            alpha = alphas.view(-1, 1, 1, 1) * torch.ones_like(a).to(self.device)
            alpha = alpha * a

            absorbing_state = torch.eye(a.size(-1), device=a.device)[0].view(1, 1, 1, -1).to(self.device)
            t[t < 10e-3] = 0.001
            betas = -torch.log(torch.tensor(t)) * self.scale
            beta_i = betas.view(-1, 1, 1, 1) * torch.ones_like(a).to(self.device)
            beta = beta_i * absorbing_state
            params = t.view(-1, 1, 1, 1) * alpha + (1 - t.view(-1, 1, 1, 1)) * beta
            params += ones
            a = Dirichlet(params).sample()

        elif self.prior == 'normal':
            alphas = -(torch.log(torch.tensor(1.0, device=a.device) - t) * self.scale)
            x_0 = torch.randn(a.size(), device=a.device)
            x_1 = a * 2 - 1
            a = alphas.view(-1, 1, 1, 1) * x_1 + (1 - alphas.view(-1, 1, 1, 1)) * x_0

        bs, n, _, d = a.shape
        a = a.permute(0, 3, 1, 2).flatten(0, 1)
        a = a.tril(-1) + a.tril(-1).transpose(1, 2)
        a = a.reshape(bs, d, n, n).permute(0, 2, 3, 1)
        return a


def get_distribution(loader, dataset):
    if os.path.isfile(f'./data/{dataset}/distributions.pt'):
        node_distrib, edge_distrib = torch.load(f'./data/{dataset}/distributions.pt')
    else:
        print('Recompute distributions')
        node_sum, edge_sum, n_node_pairs = 0, 0, 0
        for batch in loader:
            x = batch.x
            edge_attr = batch.edge_attr
            node_sum += x.sum(0)
            edge_sum += edge_attr.sum(0)
            n = batch.batch.bincount()
            n_node_pairs += (n * (n-1)).sum()
        edge_sum = torch.cat((n_node_pairs.unsqueeze(-1)-edge_sum.sum(-1), edge_sum), -1)
        node_distrib, edge_distrib = node_sum/node_sum.sum(), edge_sum/n_node_pairs
        torch.save((node_distrib, edge_distrib), f'./data/{dataset}/distributions.pt')
    return node_distrib, edge_distrib
