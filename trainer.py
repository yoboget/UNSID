
import os
import time
import copy
import wandb
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch_geometric.data import Batch

from utils.extra_feat import ExtraFeatures
from utils.noising import get_distribution, NoiserDirichlet
from utils.utils import get_num_nodes_distribution, batch_to_dense, get_conditional_input
from eval.metrics import SamplingMetrics
from models.loaders import load_denoiser, load_classifier
from logger import RunningMetric, save_model
from sample import Sampler


class Trainer:
    def __init__(self, loaders, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = 'cpu'
        print(f'Run on {self.device}')

        self.sampling = config.work_type == 'sample'
        self.work_type = config.work_type

        self.loaders = loaders
        self.extra_features = ExtraFeatures(config)
        self.prior = config.model.prior

        ### LOAD MODELS ###
        ### DENOISER ###
        if config.work_type in ['train', 'sample']:
            denoiser = load_denoiser(config, loaders['train'], self.extra_features, self.device, prior=config.model.prior,
                                     denoiser_dir=config.denoiser_dir)
            self.denoiser, self.opt, self.sched = denoiser
            self.denoiser_ema = copy.deepcopy(self.denoiser)

        if config.work_type == 'train_classifier' :
            classifier = load_classifier(config, loaders['train'], self.extra_features, self.device, train=True)
            self.classifier, self.opt_classifier, self.sched_classifier = classifier
            self.classifier_ema = copy.deepcopy(self.classifier)
        if config.work_type == 'sample' and config.conditioning.classifier_guidance:
            classifier = load_classifier(config, loaders['train'], self.extra_features, self.device, train=True)
            self.classifier, self.opt_classifier, _ = classifier

        # For comparison
        batch = next(iter(self.loaders['train']))
        nnf = batch.x.size(-1)
        nef = batch.edge_attr.size(-1)+1

        self.id = config.sampling.id
        node, edge = get_distribution(self.loaders['train'], config.dataset)

        self.cfg = config.conditioning.classifier_free_guidance
        self.cg = config.conditioning.classifier_guidance
        if self.cfg or self.cg:
            self.idx = config.conditioning.idx

        # Extract configuration variables
        self.epochs = config.training.epochs
        self.decay_iteration = config.training.decay_iteration
        self.max_num_nodes = config.data.max_num_nodes

        # Define Logger
        self.metrics = RunningMetric(['loss', 'node', 'edge'])
        self.n_logging_steps = config.log.n_loggin_steps
        self.n_logging_epochs = config.log.n_loggin_epochs
        self.best_run = {}

        self.noiser = NoiserDirichlet(config.model.T, self.prior, distributions=(node, edge), scale=config.model.scale)

        num_node_distrib = get_num_nodes_distribution(self.loaders, self.max_num_nodes, config.dataset)
        data_infos = self.max_num_nodes, nnf, nef, num_node_distrib

        if config.dataset == 'qm9_cc':
            conditional_loader = loaders['val'] if config.work_type=='train' else loaders['test']
        else:
            conditional_loader = None

        self.sampling_batch_size = config.log.sampling_batch_size
        if config.work_type in ['train', 'sample']:
            lambda_guidance = config.conditioning.lambda_guidance if config.dataset == 'qm9_cc' else None
            lambda_cfg = config.conditioning.lambda_cfg if config.dataset == 'qm9_cc' else 0.0
            self.sampler = Sampler(self.denoiser_ema, config, self.prior, self.noiser, data_infos, self.extra_features,
                                   self.sampling_batch_size, self.device, conditional_loader=conditional_loader,
                                   cfg=self.cfg, lambda_cfg=lambda_cfg, cg=self.cg, lambda_guidance=lambda_guidance)

            ref_loader = self.loaders['test'] if self.sampling else self.loaders['val']
            self.eval_samples = SamplingMetrics(config.dataset, self.max_num_nodes, self.sampling,
                                                ref_loader=ref_loader)
            self.val_size = config.log.n_val_samples

        self.dataset = config.dataset
        self.save_graphs = config.log.save_graphs


    def train(self) -> None:
        print(f'The training set contains {len(self.loaders["train"])} batches')
        starting_time = time.time(), time.process_time()
        self.step = 0
        print('Training starts...')
        for self.epoch in range(1, self.epochs + 1):

            # TRAIN
            for batch in self.loaders['train']:
                self.step += 1
                # TRAIN MODEL
                if self.work_type == 'train':
                    self.fit(batch.to(self.device), train=True)
                    if self.step % self.n_logging_steps == 0:
                        self.metrics.log(self.step, key='iter', times=starting_time)


                elif self.work_type == 'train_classifier':
                    self.fit_classifier(batch.to(self.device), train=True)
                    if self.step % self.n_logging_steps == 0:
                        self.metrics.log(self.step, key='iter', times=starting_time)
            self.metrics.log(self.step, key='train', times=starting_time)

            # VAL
            with (torch.no_grad()):
                for batch in self.loaders['val']:
                    if self.work_type == 'train':
                        self.fit(batch.to(self.device), train=False)
                    elif self.work_type == 'train_classifier':
                        self.fit_classifier(batch.to(self.device), train=False)
                val_metrics = self.metrics.log(self.step, key='val', times=starting_time)

            # SAMPLING
                if self.work_type == 'train':
                    if self.epoch % self.n_logging_epochs == 0:
                        X, A, mask, mask_adj = self.sampler(self.val_size, iter_denoising=self.id)
                        sampling_metrics = self.eval_samples(X, A, mask, mask_adj, conditioning=self.sampler.c)
                        to_save = self.denoiser_ema, self.opt, self.sched
                        self.save_model(to_save, sampling_metrics, val_metrics['loss'])
                    else:
                        to_save = self.denoiser_ema, self.opt, self.sched
                        self.save_model(to_save, None, val_metrics['loss'])
                elif self.work_type == 'train_classifier':
                    to_save = self.classifier_ema, self.opt_classifier, self.sched_classifier
                    self.save_model(to_save, None, val_metrics['loss'])

            if self.step % 20000 == 0:
                torch.save({'denoiser': self.denoiser_ema.state_dict(),
                    'optimizer': self.opt.state_dict(),
                    'scheduler': self.sched.state_dict()},
                    os.path.join(wandb.run.dir, f'best_run_{self.step}_ema.pt'))
                torch.save({'denoiser': self.denoiser.state_dict(),
                            'optimizer': self.opt.state_dict(),
                            'scheduler': self.sched.state_dict()},
                           os.path.join(wandb.run.dir, f'best_run_{self.step}.pt'))

            if self.step % self.decay_iteration == 0:
                if self.work_type == 'train':
                    self.sched.step()


    def fit(self, batch: Batch, train: bool = True):
        if train:
            self.opt.zero_grad()
            self.denoiser.train()
        else:
            self.denoiser.eval()
        # PREP
        self.X, self.A, mask = batch_to_dense(batch, max_num_nodes=self.max_num_nodes)
        diag_mask = (1-torch.eye(self.A.size(1)).to(self.A.device).unsqueeze(0)).bool()
        adj_mask = mask.unsqueeze(1) & mask.unsqueeze(2) & diag_mask

        X_noisy, A_noisy, t = self.noiser(self.X, self.A, mask)
        X_noisy, A_noisy = self.extra_features(X_noisy, A_noisy, mask)

        if self.cfg:
            X_noisy = get_conditional_input(batch.c, X_noisy, mask, self.max_num_nodes, self.device)

        if not train:
            X_pred_, A_pred_ = self.denoiser_ema(X_noisy, A_noisy, mask, t)

        else:
            X_pred_, A_pred_ = self.denoiser(X_noisy, A_noisy, mask, t)

        X_pred = X_pred_[mask]
        X_targ = self.X[mask]

        A_pred = A_pred_[adj_mask]
        A_targ = self.A[adj_mask]

        if X_targ.size(-1) > 1:
            loss_x = F.cross_entropy(X_pred, X_targ)
            n, m = X_targ.size(0), A_targ.size(0)

            loss_a = F.cross_entropy(A_pred, A_targ)
            loss = (n / (m + n)) * loss_x + (m / (m + n)) * loss_a

        else:
            A_targ = A_targ[..., 1:]
            loss_a = F.binary_cross_entropy(A_pred.sigmoid(), A_targ)
            loss = loss_a
            loss_x = torch.zeros(1, dtype=X_targ.dtype, device=X_targ.device)

        if train:
            loss.backward()
            self.opt.step()
            update_ema(self.denoiser, self.denoiser_ema)

        to_log = [loss.item(), loss_x.item(), loss_a.item()]
        self.metrics.step(to_log, train)

    def fit_classifier(self, batch: Batch, train: bool = True):
        if train:
            self.opt_classifier.zero_grad()
            self.classifier.train()
        else:
            self.classifier.eval()

        # PREP
        self.X, self.A, mask = batch_to_dense(batch, max_num_nodes=self.max_num_nodes)

        X_noisy, A_noisy, t = self.noiser(self.X, self.A, mask)
        # X_noisy, A_noisy = self.extra_features(X_noisy, A_noisy, mask)

        if not train:
            pred, _ = self.classifier_ema(X_noisy, A_noisy, mask, t)

        else:
            pred, _ = self.classifier(X_noisy, A_noisy, mask, t)

        loss = F.mse_loss(pred, batch.c[..., self.idx])
        if train:
            loss.backward()
            self.opt_classifier.step()
            update_ema(self.classifier, self.classifier_ema)

        to_log = [loss.item()]
        self.metrics.step(to_log, train)

    def fit_node_predictor(self, batch: Batch, train: bool = True):
        if train:
            self.opt_node_predictor.zero_grad()
            self.node_predictor.train()
        else:
            self.node_predictor.eval()



        if not train:
            pred, _ = self.node_predictor_ema(batch.c)

        else:
            pred, _ = self.node_predictor(X_noisy, A_noisy, mask)

        loss = F.mse_loss(pred, batch.c[..., self.idx])
        if train:
            loss.backward()
            self.opt_classifier.step()
            update_ema(self.classifier, self.classifier_ema)

        to_log = [loss.item()]
        self.metrics.step(to_log, train)


    def get_reconstruction_loss(self, x_target, edge_target, x_pred, edge_attr_pred, masks, n, m):
        mask, edge_mask = masks
        x_pred = x_pred * mask.unsqueeze(-1)
        loss_x = F.cross_entropy(x_pred.permute(0, 2, 1), x_target, reduction='none')
        loss_x = (loss_x * mask).mean()
        loss_edge = F.cross_entropy(edge_attr_pred.permute(0, 3, 1, 2), edge_target, reduction='none')
        loss_edge = (loss_edge * edge_mask).mean()
        rec_loss = (n / (m + n)) * loss_x + (m / (m + n)) * loss_edge
        return rec_loss, loss_x, loss_edge

    def save_model(self,  to_save, metrics, loss):
        if metrics is not None:
            if self.dataset in ['zinc', 'qm9', 'qm9_cc', 'qm9_dg', 'qm9H']:
                ref_metric_name = 'nspdk'
                ref_metric_name_save = 'nspdk'
                self.best_run = save_model(metrics[ref_metric_name], best_run=self.best_run,
                                           to_save=to_save, step=self.step, save_name=ref_metric_name_save)
                ref_metric_name = 'fcd'
                ref_metric_name_save = 'fcd'
                self.best_run = save_model(metrics[ref_metric_name], best_run=self.best_run,
                                           to_save=to_save, step=self.step, save_name=ref_metric_name_save)
            else:
                ref_metric_name = 'avg'
                ref_metric_name_save = 'avg'
                self.best_run = save_model(metrics[ref_metric_name], best_run=self.best_run,
                                           to_save=to_save, step=self.step, save_name=ref_metric_name_save)
                ref_metric_name = 'valid'
                ref_metric_name_save = 'valid'
                if 'valid' in metrics.keys():
                    self.best_run = save_model(metrics[ref_metric_name], best_run=self.best_run,
                                               to_save=to_save, step=self.step,
                                               save_name=ref_metric_name_save, minimize=False)

        self.best_run = save_model(loss, best_run=self.best_run, to_save=to_save,
                                   step=self.step, save_name='loss')

    def compute_conditional_loss(self, X, A, mask, cond):
        X_ = Categorical(X.softmax(-1)).sample().unsqueeze(-1)
        X_ = torch.zeros_like(X).scatter_(-1, X_, 1)
        X = X + X_ - X.detach()

        A_ = Categorical(A.softmax(-1)).sample().unsqueeze(-1)
        A_ = torch.zeros_like(A).scatter_(-1, A_, 1)
        A = A + A_ - A.detach()

        A = 0.5 * (A + A.transpose(1, 2))
        X, A = self.extra_features(X, A, mask)
        y_pred, _ = self.regressor(X, A, mask)
        return F.mse_loss(y_pred, self.cond[:, 0])



@torch.no_grad()
def update_ema(model, ema_model, ema_decay=0.999):
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = ema_decay * ema_param.data + (1.0 - ema_decay) * model_param.data
    # Buffer copy for Batch_Norm modules
    for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data = model_buffer.data.clone()

