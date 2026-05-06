import os
import torch
from torch import optim
from models.nn import Mlp
from models.models import DenseGNN, NodePredictor
from models.digress_model import GraphTransformer



def load_denoiser(config, loader, extra_features, device, prior,
                  denoiser_dir=None):
    """
    Loads the denoiser and critic models, along with their optimizers and schedulers.

    Args:
        config: Configuration object.
        loader: Data loader containing training data.
        extra_features: Object containing extra features information.
        device: Device to load models onto (e.g., 'cuda' or 'cpu').
        prior: Type of diffusion prior ('masking', 'absorbing', 'marginal').
        denoiser_dir: Directory to load pre-trained denoiser model from (optional).
        critic_dir: Directory to load pre-trained critic model from (optional).

    Returns:
        Tuple: ((denoiser, optimizer, scheduler), (critic, critic_optimizer, critic_scheduler))
    """
    sizes  = get_input_sizes(config, loader, extra_features)
    input_sizes, output_sizes, hidden_size, n_node_attr =  sizes

    # --- Denoiser ---
    print("Loading Denoiser...")
    if config.model.architecture == 'digress':
        denoiser = GraphTransformer(config, *input_sizes, *output_sizes, hidden_size).to(device)
    else:
        denoiser = DenseGNN(config, *input_sizes, *output_sizes, hidden_size,
                            norm_out=True).to(device)

    params = list(denoiser.parameters())
    betas = config.training.betas.beta1, config.training.betas.beta2
    # opt = optim.Adam(params, lr=config.training.learning_rate, betas=betas)
    opt = torch.optim.AdamW(params, lr=config.training.learning_rate, amsgrad=True, weight_decay=1.0e-12)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, config.training.lr_decay)
    n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f'Number of parameters in the encoder: {n_params}')

    if denoiser_dir is not None:
        # filename = 'best_run_avg_ema.pt'
        if config.dataset in ['qm9', 'qm9_cc', 'zinc', 'qm9H']:
            filename = 'best_run_fcd.pt'
        else:
            filename = f'best_run_{config.sampling.ckp}.pt'
        loaded = load_trained_model(denoiser, opt, scheduler,'denoiser', denoiser_dir, filename, device)
        denoiser, opt, scheduler = loaded

    return denoiser, opt, scheduler


def load_trained_model(model, opt, scheduler, model_name, model_dir, filename, device):
    model_path = os.path.join(model_dir, filename)
    saved_model = torch.load(model_path, map_location=device)
    model.load_state_dict(saved_model[model_name])
    if opt is not None:
        opt.load_state_dict(saved_model['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(saved_model['scheduler'])
    return model, opt, scheduler

def get_input_sizes(config, loader, extra_features):
    n_extra_feat = extra_features.n_features
    batch = next(iter(loader))
    n_node_attr = batch.x.size(-1)
    n_edge_attr = batch.edge_attr.size(1)
    nhf = config.model.nhf

    print(f"Node features: {n_node_attr}, Extra features: {n_extra_feat}")
    time_feat = 2
    nnf_in = n_node_attr + n_extra_feat + time_feat
    nef_in = n_edge_attr + 1
    if extra_features.rrwp:
        print(extra_features.rrwp)
        nef_in += 5
    if extra_features.discrete_features:
        nnf_in += n_node_attr
        nef_in += n_edge_attr + 1
    if config.conditioning.classifier_free_guidance:
        nnf_in += 4

    nnf_out = n_node_attr
    nef_out = n_edge_attr
    if n_node_attr > 1:
        nef_out += 1
    return (nnf_in, nef_in), (nnf_out, nef_out), nhf, n_node_attr



def load_classifier(config, loader, extra_features, device, train=True):
    (nnf_in, nef_in), _, nhf, n_node_attr = get_input_sizes(config, loader, extra_features)

    dc = len(config.conditioning.idx)
    # dc = 3
    nnf_in -= extra_features.n_features
    if config.conditioning.classifier_free_guidance:
        nnf_in -= 4

    nhf = 128
    config_ = config.copy()
    from easydict import EasyDict as edict
    config_ = edict(config_)
    config_.model.n_layers = 4
    classifier = DenseGNN(config_, nnf_in, nef_in, dc, 1, nhf, norm_out=True, global_pooling=True).to(device)
    params = list(classifier.parameters())
    betas = config.training.betas.beta1, config.training.betas.beta2
    opt_classifier = optim.Adam(params, lr=config.training.learning_rate, betas=betas)
    scheduler_classifier = optim.lr_scheduler.ExponentialLR(opt_classifier, config.training.lr_decay)
    if not train:
        idx = config.conditioning.idx
        print(idx)
        filename = f'classifier{idx}.pt'
        model_dir = './pretrained_models/'
        classifier, opt_classifier, scheduler_classifier = load_trained_model(classifier, None, None, 'denoiser', model_dir, filename, device)
    return classifier, opt_classifier, scheduler_classifier

def load_node_predictor(config, nhf, device, model_dir):
    NF_IN = 3
    nf_out = config.data.max_num_nodes
    node_predictor = Mlp(NF_IN, nf_out, [nhf, nhf]).to(device)
    # node_predictor = NodePredictor(NF_IN, nf_out, [nhf, nhf], 5).to(device)
    params = list(node_predictor.parameters())
    betas = config.training.betas.beta1, config.training.betas.beta2
    opt_node_predictor = optim.Adam(params, lr=config.training.learning_rate, betas=betas)
    scheduler_node_predictor = optim.lr_scheduler.ExponentialLR(opt_node_predictor, config.training.lr_decay)
    if model_dir is not None:
        filename = 'best_run_loss.pt'
        node_predictor = load_trained_model(node_predictor, 'denoiser', model_dir, filename, device)
    return node_predictor, opt_node_predictor, scheduler_node_predictor

def load_pretrained_node_predictor(max_num_nodes, nhf, device, model_dir, idx=None):
    # nf_in = 3 if idx == '' else len(idx)
    # nf_in = len(idx)
    nf_in = 3
    nf_out = max_num_nodes
    node_predictor = Mlp(nf_in, nf_out, [nhf, nhf]).to(device)
    if model_dir is not None:
        # filename = f'node_predictor{idx}.pt'
        filename = f'node_predictor.pt'
        node_predictor = load_trained_model(node_predictor, None, None,  'denoiser', model_dir,
                                            filename, device)
    return node_predictor