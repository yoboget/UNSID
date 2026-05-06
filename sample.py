import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet

from models.loaders import load_pretrained_node_predictor, load_classifier
from utils.utils import get_conditional_input
from torch.distributions.categorical import Categorical
from utils.flow_matching import DirichletConditionalFlow


class Sampler:
    def __init__(self, denoiser, config, prior, noiser, data_infos, extra_features, sampling_batch_size, device,
                 conditional_loader=None, cfg=False, lambda_cfg=1.0, cg=False, lambda_guidance=2.0):

        self.denoiser = denoiser
        self.device = device
        self.max_num_nodes, self.n_node_attr, self.n_edge_attr, self.num_node_distribution = data_infos
        self.eye_x = torch.eye(self.n_node_attr, device=self.device)
        self.eye_a = torch.eye(self.n_edge_attr, device=self.device)
        self.prior = prior
        self.noiser = noiser
        self.scale = noiser.scale
        self.extra_features = extra_features
        self.T = noiser.T
        self.sampling_batch_size = sampling_batch_size
        self.alpha_node = torch.tensor([config.sampling.node_alpha], device=self.device)
        self.alpha_edge = torch.tensor([config.sampling.edge_alpha], device=self.device)

        self.flow_x = DirichletConditionalFlow(K=self.n_node_attr, T=self.T, scale=self.scale)
        self.flow_a = DirichletConditionalFlow(K=self.n_edge_attr, T=self.T, scale=self.scale)

        self.cfg = cfg
        self.lambda_cfg = lambda_cfg
        self.classifier_guidance = cg

        if conditional_loader is not None:
            # Load conditional feature batch
            batch = next(iter(conditional_loader))
            idx = config.conditioning.idx
            self.c = batch.c.to(self.device)[..., idx]
            self.n = batch.batch.bincount().to(device=self.device)
            NHF = 128
            MODEL_DIR = './pretrained_models/'
            # if self.n is None:
            # self.node_predictor, _, _ = load_pretrained_node_predictor(self.max_num_nodes, NHF, device, MODEL_DIR, idx)
            if cg:
                self.classifier, _, _ = load_classifier(config, conditional_loader, extra_features, device, train=False)
                self.lamda_guidance = lambda_guidance

        else:
            self.c = None

    def __call__(self, n_samples, iter_denoising=True):
        self.denoiser.eval()
        batch_size = self.sampling_batch_size if self.sampling_batch_size < n_samples else n_samples
        x, a, mask, mask_adj = self.sample_batch(batch_size,iter_denoising=iter_denoising)
        remaining_samples = n_samples - batch_size

        while remaining_samples > 0:
            x_, a_, mask_, mask_adj_ = self.sample_batch(batch_size, iter_denoising=iter_denoising)
            x, a, mask, mask_adj = (torch.cat((x , x_), dim=0), torch.cat((a , a_), dim=0),
                                torch.cat((mask , mask_), dim=0), torch.cat((mask_adj, mask_adj_), dim=0))
            remaining_samples -= batch_size
        X, A, mask, mask_adj = x[:n_samples], a[:n_samples], mask[:n_samples], mask_adj[:n_samples]
        return X, A, mask, mask_adj


    def sample_batch(self, n_samples, iter_denoising=True):
        print('Sampling starts... ')
        if self.cfg or self.classifier_guidance:
            n = self.n
            # n_probs = self.node_predictor(self.c)
            # n = Categorical(probs=n_probs.softmax(-1)).sample().to(self.device).squeeze() + 1
        else:
            n = Categorical(probs=self.num_node_distribution).sample((n_samples,)).to(self.device).squeeze()
        mask = torch.tril(torch.ones(self.max_num_nodes, self.max_num_nodes)).to(self.device)[n-1].bool()
        x_t, a_t, mask, mask_adj = self.sample_noise(n_samples, mask, iter_denoising)

        start_time = time.time()
        elaps_list = []
        if iter_denoising:
            for t in range(self.T):
                x_t, a_t = self.iterative_denoising_step(t, x_t, a_t, mask, mask_adj)
                if (t + 1) % 20 == 0:
                    print(f'{t + 1} timesteps done. Sampling resumes...')

                if (t % 100) == 0:
                    elaps = time.time() - start_time
                    print(elaps)
                    elaps_list.append(elaps)
                    start_time = time.time()
        else:
            t_span = torch.linspace(1, self.flow_x.alpha_max.item(), self.T+1, device=self.device)
            for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
                x_t, a_t = self.flow_matching_step(i, x_t, a_t, mask, mask_adj)
                if (i + 1) % 20 == 0:
                    print(f'{i + 1} timesteps done. Sampling resumes...')
                if (i % 100) == 0:
                    elaps = time.time() - start_time
                    print(elaps)
                    elaps_list.append(elaps)
                    start_time = time.time()
        print('Sampling done.')
        time_array = np.asarray([elaps_list[1:6]])
        print('time mean and std : ', time_array.mean(), time_array.std())
        return x_t, a_t, mask.bool(), mask_adj.bool().squeeze()


    def flow_matching_step(self, i, x_t, a_t, mask, mask_adj):

        x_t = x_t[..., :self.n_node_attr]
        a_t = a_t[..., :self.n_edge_attr]
        # print(a_t[0, 0])

        t_ = (i+1)/self.T * torch.ones(x_t.size(0), x_t.size(1)).to(self.device)
        x_t = torch.cat((x_t, t_[..., None], 1 - t_[..., None]), dim=-1)
        s = 1-(torch.log(torch.tensor([1.0], device=x_t.device) - i/self.T) * self.scale)
        t = 1 - (torch.log(torch.tensor([1.0], device=x_t.device) - (i + 1) / self.T) * self.scale)

        x_t, a_t = self.extra_features(x_t, a_t, mask)
        p_x, p_a = self.denoiser(x_t, a_t, mask.bool(), t_[0, 0].repeat(x_t.size(0), 1))
        if self.n_edge_attr <= 2:
            p_a = torch.sigmoid(p_a)
            p_a = torch.cat((1-p_a, p_a), dim=-1)
            a_t = a_t[..., :self.n_edge_attr]
        else:
            p_x = torch.softmax(p_x, dim=-1)
            p_a = torch.softmax(p_a, dim=-1)
            x_t = x_t[..., :self.n_node_attr]
            a_t = a_t[..., :self.n_edge_attr]

        if self.n_node_attr > 2:
            c_factor_x = self.flow_x.c_factor(x_t.cpu().numpy(), s.item())
            c_factor_x = torch.from_numpy(c_factor_x).to(x_t)
            cond_flows_x = (self.eye_x - x_t.unsqueeze(-1)) * c_factor_x.unsqueeze(-2)
            flow_x = (p_x.unsqueeze(-2) * cond_flows_x).sum(-1)
            x_t = x_t + flow_x * (t - s)

        c_factor_a = self.flow_a.c_factor(a_t.cpu().numpy(), s.item())
        c_factor_a = torch.from_numpy(c_factor_a).to(a_t)
        cond_flows_a = (self.eye_a - a_t.unsqueeze(-1)) * c_factor_a.unsqueeze(-2)
        flow_a = (p_a.unsqueeze(-2) * cond_flows_a).sum(-1)
        a_t = a_t + flow_a * (t - s)

        if (i+1) == self.T:
            if self.n_node_attr == 1:
                x_0 = torch.ones(*x_t.shape[:-1], 1).to(self.device) * mask.unsqueeze(-1)
            else:
                x_0 = p_x.argmax(-1) * mask
            a_0 = p_a.argmax(-1) * mask_adj.squeeze()
            return x_0.int() , a_0.int()
        else:
            return x_t * mask.unsqueeze(-1), a_t * mask_adj

    def iterative_denoising_step(self, t, x_t, a_t, mask, mask_adj):
        """
            Inspired from Digress diffusion_model_discrete.py
            Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well
           """
        self.t = t + 1

        if self.classifier_guidance:
            grad_x, grad_a = self.get_classifier_gradiant(x_t, a_t, mask)
            logit_x = torch.log(x_t[..., :self.n_node_attr])
            logit_a = torch.log(a_t[..., :self.n_edge_attr])
            logit_x[~mask.squeeze().bool()] = -1e9
            logit_a[~mask_adj.squeeze().bool()] = -1e9
            # n, m = mask.float().sum([1]), mask_adj.sum([1, 2, 3])
            # (n / (n + m)).view(-1, 1, 1) *
            logit_x = logit_x - self.lamda_guidance * grad_x[..., :self.n_node_attr]
            logit_a = logit_a - self.lamda_guidance * grad_a[..., :self.n_edge_attr]
            x_t[..., :self.n_node_attr] = torch.softmax(logit_x, dim=-1) * mask.unsqueeze(-1)
            a_t[..., :self.n_edge_attr] = torch.softmax(logit_a, dim=-1) * mask_adj

        #### DENOISING ###
        x_0, a_0 = self.denoising_step(x_t, a_t, mask, mask_adj, t)

        if self.n_node_attr == 1:
            x_0 = x_t[..., :-2]

        assert (a_0 != a_0.transpose(1, 2)).sum() == 0, (a_0 != a_0.transpose(1, 2)).sum()

        #### RE-NOISING ###
        if self.alpha_edge > 1 or self.alpha_node > 1:
            t_node = 1-(-(self.alpha_node-1)/self.noiser.scale).exp()
            t_node = t_node + (t / self.T) * (1 - t_node)

            t_edge = 1 - (-(self.alpha_edge-1) / self.noiser.scale).exp()
            t_edge = t_edge + (t / self.T) * (1 - t_edge)
            x_s, a_s, _ = self.noiser(x_0, a_0, mask, t=torch.tensor([t / self.T]).float(),
                                   t_node=t_node, t_edge=t_edge)
        else:
            x_s, a_s, _ = self.noiser(x_0, a_0, mask, t=torch.tensor([t/ self.T]).float())

        if self.t == self.T:
            x_0 = x_0.argmax(-1) * mask
            a_0 = a_0.argmax(-1) * mask_adj.squeeze()
            return x_0.int(), a_0.int()
        else:
            return x_s * mask.unsqueeze(-1), a_s * mask_adj


    def denoising_step(self, x_t, a_t, mask, mask_adj, t):
        x_t, a_t = self.extra_features(x_t, a_t, mask)
        t = torch.tensor([t], device=self.device).float().repeat(x_t.size(0), 1)
        if self.cfg:
            x_t_ = get_conditional_input(self.c, x_t, mask, self.max_num_nodes, self.device,
                                        training=False, conditional=False)
            logit_x_uncond, logit_a_uncond = self.denoiser(x_t_, a_t, mask.bool(), t/self.T)

            x_t_= get_conditional_input(self.c, x_t, mask, self.max_num_nodes, self.device,
                                  training=False, conditional=True)
            logit_x_cond, logit_a_cond = self.denoiser(x_t_, a_t, mask.bool(), t/self.T)

            logit_x = self.lambda_cfg * logit_x_cond + (1 - self.lambda_cfg) * logit_x_uncond
            logit_a = self.lambda_cfg * logit_a_cond + (1 - self.lambda_cfg) * logit_a_uncond

        else:
            logit_x, logit_a = self.denoiser(x_t, a_t, mask.bool(), t/self.T)

        p_x = torch.softmax(logit_x, dim=-1)
        if self.n_edge_attr > 2:
            p_a = torch.softmax(logit_a, dim=-1)
        else:
            p_a = torch.sigmoid(logit_a)
            p_a = torch.cat((1-p_a, p_a), dim=-1)

        x = Categorical(probs=p_x).sample().to(self.device)
        x_0 = F.one_hot(x, num_classes=p_x.shape[-1]).float()
        x_0 = x_0 * mask.unsqueeze(-1)

        a = Categorical(probs=p_a).sample().to(self.device)
        a = F.one_hot(a, num_classes=p_a.shape[-1]).float()
        a = a.permute(0, 3, 1, 2)
        a = a.tril(-1) + a.tril(-1).transpose(2, 3)
        a_0 = a.permute(0, 2, 3, 1)
        a_0 = a_0 * mask_adj
        return x_0, a_0


    def sample_noise(self, n_samples, mask, iter_denoising):
        if self.n_node_attr > 1:
            if self.prior == 'dirch':
                if self.alpha_node > 1:
                    ones = torch.ones(n_samples, self.max_num_nodes,
                                      self.n_node_attr, device=self.device)
                    marg_sample = Categorical(self.noiser.marginal_x).sample((n_samples,
                                                                              self.max_num_nodes)).to(self.device)
                    marg_sample = F.one_hot(marg_sample, num_classes=self.n_node_attr).float()
                    marg_sample = marg_sample * (self.alpha_node-1)
                    x_t = Dirichlet(marg_sample + ones).sample()
                else:
                    x_t = Dirichlet(torch.ones(n_samples, self.max_num_nodes,
                                           self.n_node_attr, device=self.device)).sample()
            elif self.prior == 'dirch_k':
                x_t = Dirichlet(torch.ones(n_samples, self.max_num_nodes,
                                           self.n_node_attr, device=self.device)).sample()
            elif self.prior in ['dirch_marg', 'dirch_marg_t']:
                ones = torch.ones(n_samples, self.max_num_nodes,
                                           self.n_node_attr, device=self.device)
                marg_sample = Categorical(self.noiser.marginal_x).sample((n_samples,
                                                                          self.max_num_nodes)).to(self.device)
                marg_sample = F.one_hot(marg_sample, num_classes=self.n_node_attr).float()
                marg_sample = marg_sample * (self.n_node_attr - 2)
                x_t = Dirichlet(marg_sample+ones).sample()
            elif self.prior == 'normal':
                x_t = torch.randn(n_samples, self.max_num_nodes, self.n_node_attr,
                                  device=self.device)


        else:
            x_t = torch.ones(n_samples, self.max_num_nodes, 1, device=self.device) * mask.unsqueeze(-1)
        t = torch.ones(n_samples, self.max_num_nodes, 1, device=self.device)
        x_t = torch.cat((x_t, 1-t, t), dim=-1) * mask.unsqueeze(-1)

        if self.prior == 'dirch':
            if self.alpha_edge > 1:
                ones = torch.ones(n_samples, self.max_num_nodes, self.max_num_nodes,
                                  self.n_edge_attr, device=self.device)
                marg_sample = Categorical(self.noiser.marginal_a).sample((n_samples, self.max_num_nodes,
                                                                          self.max_num_nodes)).to(self.device)
                marg_sample = F.one_hot(marg_sample, num_classes=self.n_edge_attr).float()
                marg_sample = marg_sample * (self.alpha_edge - 1)
                a_t = Dirichlet(marg_sample + ones).sample()
            else:
                a_t = Dirichlet(torch.ones(n_samples, self.max_num_nodes,
                                   self.max_num_nodes, self.n_edge_attr, device=self.device)).sample()
        elif self.prior == 'dirch_k':
            a_t = Dirichlet(torch.ones(n_samples, self.max_num_nodes,self.max_num_nodes, self.n_edge_attr,
                                       device=self.device)).sample()
        elif self.prior in ['dirch_marg', 'dirch_marg_t', 'dirch_mix']:
            ones = torch.ones(n_samples, self.max_num_nodes, self.max_num_nodes,
                              self.n_edge_attr, device=self.device)
            marg_sample = Categorical(self.noiser.marginal_a).sample((n_samples, self.max_num_nodes,
                                                                      self.max_num_nodes)).to(self.device)
            marg_sample = F.one_hot(marg_sample, num_classes=self.n_edge_attr).float()
            k = self.n_edge_attr if self.n_edge_attr > 2 else 3
            marg_sample = marg_sample * (2 - 2)
            a_t = Dirichlet(marg_sample + ones).sample()
        elif self.prior == 'dirch2':
            ones = torch.ones(n_samples, self.max_num_nodes, self.max_num_nodes, 1, device=self.device)
            params = ones * self.noiser.base_params.view(1, 1, 1, -1).to(self.device)
            a_t = Dirichlet(params).sample()

        elif self.prior == 'abs':
            ones = torch.ones(n_samples, self.max_num_nodes, self.max_num_nodes, 1, device=self.device)
            absorbing_state = torch.eye(self.n_edge_attr, device=self.device)[0].view(1, 1, 1, -1).to(self.device)
            params = -torch.log(torch.tensor(1/self.T)) * self.scale * absorbing_state
            params = ones + params
            a_t = Dirichlet(params).sample()

        elif self.prior == 'normal':
            a_t = torch.randn(n_samples, self.max_num_nodes,self.max_num_nodes, self.n_edge_attr, device=self.device)
        bs, n, _, d = a_t.shape
        a_t = a_t.permute(0, 3, 1, 2).flatten(0, 1)
        a_t = a_t.tril(-1) + a_t.tril(-1).transpose(1, 2)
        a_t = a_t.reshape(bs, d, n, n).permute(0, 2, 3, 1)
        diag = (1 - torch.eye(a_t.shape[-2]).to(self.device)).unsqueeze(0).unsqueeze(-1)
        mask_adj = mask[..., None, None] * mask[:, None, :, None] * diag
        a_t = a_t * mask_adj
        return x_t, a_t, mask, mask_adj

    def get_classifier_gradiant(self, x, a, mask):
        X, A = x.clone().detach(), a.clone().detach()
        X.requires_grad_(True), A.requires_grad_(True)
        with torch.enable_grad():
            # Forward pass
            output, _ = self.classifier(X, A, mask.bool())
            # If target_output is given, compute loss-based gradients
            loss = F.mse_loss(output, self.c, reduction='sum')
            loss.backward()
        return X.grad, A.grad
