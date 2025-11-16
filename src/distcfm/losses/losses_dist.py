import torch
import torch.nn as nn

def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

def replicate_fn(n: int, m: int, x: torch.Tensor) -> torch.Tensor:
    data_shape = x.shape[1:]
    x = x.reshape(n, 1, *data_shape)
    x = x.repeat(1, m, *([1] * len(data_shape)))
    x = x.reshape(n * m, *data_shape)
    return x

def split_fn(n: int, m: int, x: torch.Tensor) -> torch.Tensor:
    data_shape = x.shape[1:]
    x = x.reshape(n, m, *data_shape)
    return x

def compute_rho_diagonal_fn(x, y, cfg):
    """
    Args:
        x: (B, M, D1, ..)
        y: (B, M, D1, ..)
    Returns:
        (B, M)
    """
    x = x.reshape(x.shape[0], x.shape[1], -1)
    y = y.reshape(y.shape[0], y.shape[1], -1)
    eps = 1e-8

    if cfg.loss.rho == "norm":
        confinement_term = torch.norm(x - y + eps, p=2, dim=-1) ** cfg.loss.beta
    elif cfg.loss.rho == "rbf":
        confinement_term = -torch.exp(-(torch.norm(x - y + eps, p=2, dim=-1) ** 2) / (2 * cfg.loss.rbf_sigma**2))
    elif cfg.loss.rho == "imq":
        confinement_term = -1 / (torch.sqrt(cfg.loss.imq_c + torch.norm(x - y + eps, p=2, dim=-1)) ** 2)
    elif cfg.loss.rho == "exp":
        confinement_term = -torch.exp(-torch.norm(x - y + eps, p=2, dim=-1) / cfg.loss.exp_sigma)
    else:
        raise ValueError(f"Unknown rho_diagonal: {cfg.loss.rho}")
    return confinement_term

def compute_rho_fn(x, y, cfg):
    x = x.reshape(x.shape[0], x.shape[1], -1)
    y = y.reshape(y.shape[0], y.shape[1], -1)
    eps = 1e-8
    diff = x[:, :, None, :] - y[:, None, :, :]  # (B, M, M, D)
    if cfg.loss.rho == "norm":
        interaction_term = torch.norm(diff + eps, p=2, dim=-1) ** cfg.loss.beta
    elif cfg.loss.rho == "rbf":
        interaction_term = -torch.exp(-(torch.norm(diff + eps, p=2, dim=-1) ** 2) / (2 * cfg.loss.rbf_sigma**2))
    elif cfg.loss.rho == "imq":
        interaction_term = -1 / (torch.sqrt(cfg.loss.imq_c + torch.norm(diff + eps, p=2, dim=-1) ** 2))
    elif cfg.loss.rho == "exp":
        interaction_term = -torch.exp(-torch.norm(diff + eps, p=2, dim=-1) / cfg.loss.exp_sigma)
    else:
        raise ValueError(f"Unknown rho: {cfg.loss.rho}")
    interaction_term = interaction_term - torch.eye(cfg.loss.m, device=interaction_term.device) * interaction_term
    return interaction_term


def get_loss_weighting_fn(cfg):
    if cfg.loss.loss_weighting == "sigmoid":
        def loss_weighting_fn(alpha, sigma):
            beta = cfg.loss.loss_weighting_beta
            w = (1 + torch.exp(beta - torch.log((alpha**2) / (sigma**2)))) ** (-1)
            return w
    elif cfg.loss.loss_weighting == "constant":
        def loss_weighting_fn(alpha, sigma):
            return torch.ones_like(alpha)
    else:
        raise ValueError(f"Unknown loss weighting: {cfg.loss.loss_weighting}")

    return loss_weighting_fn


def get_distributional_loss_fn(cfg, sde):
    loss_weighting_fn = get_loss_weighting_fn(cfg)
    def loss_fn(model, x, step):
        device = x.device
        N, data_shape = x.shape[0], x.shape[1:]
        t_cond = torch.rand(N, device=device) * cfg.sde.t_max  # [B]
        alpha_t_cond, var_t_cond = sde.get_coefficients(t_cond) # Shape: [B,]
        sigma_t_cond = torch.sqrt(var_t_cond)
        w = loss_weighting_fn(alpha_t_cond, sigma_t_cond)

        alpha_t_cond, var_t_cond = broadcast_to_shape(alpha_t_cond, x.shape), broadcast_to_shape(var_t_cond, x.shape)

        noise_cond = torch.randn_like(x, device=x.device)
        xt = alpha_t_cond * x + torch.sqrt(var_t_cond) * noise_cond

        # Replicate for population: [B * M, **data_shape]
        x0_population = replicate_fn(n=N, m=cfg.loss.m, x=x)
        t_population = replicate_fn(n=N, m=cfg.loss.m, x=t_cond)
        xt_population = replicate_fn(n=N, m=cfg.loss.m, x=xt)

        # sample exogenous noise of that shape
        noise_population = torch.randn_like(x0_population, device=device)

        # output population
        output_population = model(xt_population, t_population, noise_population)

        # split the output into [B, M, **data_shape]
        x0_population = split_fn(n=N, m=cfg.loss.m, x=x0_population)
        output_population = split_fn(n=N, m=cfg.loss.m, x=output_population)
        
        confinement_term = compute_rho_diagonal_fn(x0_population, output_population, cfg)  # B, M
        interaction_term = compute_rho_fn(output_population, output_population, cfg)  # B, M, M

        interaction_term = -(cfg.loss.lambda_ / (2 * (cfg.loss.m - 1))) * torch.sum(interaction_term, axis=-1)  # B, M

        # compute seperate components
        interaction_term = torch.mean(interaction_term, axis=-1)  # B
        confinement_term = torch.mean(confinement_term, axis=-1)  # B

        # temporary loss weighting
        weighted_confinement_term = confinement_term * w  # (B,)
        weighted_interaction_term = interaction_term * w  # (B,)

        weighted_interaction_term = torch.mean(weighted_interaction_term)
        weighted_confinement_term = torch.mean(weighted_confinement_term)
        return {"interaction_loss": weighted_interaction_term, "confinement_loss": weighted_confinement_term,}, {}

    return loss_fn