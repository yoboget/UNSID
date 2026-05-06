import torch

import wandb
import yaml
from dataset import get_dataset
from easydict import EasyDict as edict
from trainer import Trainer

from utils.parser import parse_args
from utils.utils import print_eval

def get_dense_batch(batch, dataset='qm9'):
    from torch_geometric.utils import to_dense_batch, to_dense_adj
    x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
    x, mask = to_dense_batch(x, batch_idx)
    if dataset == 'qm9':
        x = x[..., :4].argmax(-1)
    elif dataset in ['qm9H', 'qm9_cc']:
        x = x[..., :5].argmax(-1)
    elif dataset == 'zinc':
        x = x[..., :9].argmax(-1)
    adj = to_dense_adj(edge_index, batch_idx, edge_attr)
    adj_ = adj.argmax(-1) + 1
    adj_[adj.sum(-1) == 0] = 0
    mask_adj = mask.unsqueeze(-1) * mask.unsqueeze(-2)

    return x, adj_, mask, mask_adj


def main() -> None:
    # Parse command line arguments
    args = parse_args()
    work_type = args.work_type
    dataset = args.dataset

    if args.wandb == 'no':
        args.wandb = 'disabled'
    elif args.wandb == 'on':
        args.wandb = 'online'
    elif args.wandb == 'off':
        args.wandb = 'offline'

    config_path = f'./config/{dataset}.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    config = edict(config)
    config = update_config(config, dataset, work_type, args)

    if work_type == 'train':
        config.denoiser_dir = args.denoiser_dir
        wandb.init(project=f'Unsid_{dataset}', config=config, mode=args.wandb)
        dataloader = get_dataset(config)
        trainer = Trainer(dataloader, config)
        trainer.train()

    elif  work_type == 'train_classifier':
        wandb.init(project=f'Unsid_{dataset}_classifer', config=config, mode=args.wandb)
        dataloader = get_dataset(config)
        trainer = Trainer(dataloader, config)
        trainer.train()

    elif work_type == 'sample':
        runs = []
        N_RUNS = 3
        config.denoiser_dir = args.denoiser_dir
        wandb.init(project=f'Unsid_{dataset}_sample', config=config, mode=args.wandb)
        for r in range(N_RUNS):
        # for T in [16, 32, 64, 128, 256, 512, 1024]:
        #     config.model.T = T
            wandb.init(project=f'Dirisid_{dataset}_sample', config=config, mode=args.wandb)
            dataloader = get_dataset(config)
            trainer = Trainer(dataloader, config)
            with torch.no_grad():
                # X, A, mask, mask_adj = get_dense_batch(next(iter(dataloader['train'])), dataset='zinc') ## For train stats
                X, A, mask, mask_adj= trainer.sampler(config.log.n_samples_generation, iter_denoising=config.sampling.id)
                run = trainer.eval_samples(X, A, mask, mask_adj, conditioning=trainer.sampler.c)
            runs.append(run)
            # wandb.finish()

        print_eval(runs, dataset)
    wandb.finish()


def update_config(config, dataset, work_type, args):
    config.dataset = dataset
    config.work_type = work_type
    config.denoiser_dir = args.denoiser_dir
    return config


if __name__ == "__main__":
    main()

