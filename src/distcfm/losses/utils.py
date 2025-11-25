import torch

def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

def l2_loss(pred, target, weighting, stop_gradient=False,):
    """Computes the mean squared L2 loss."""
    if stop_gradient:
        actual_target = torch.detach(target)
    else:
        actual_target = target
    delta = pred - actual_target
    delta_sq = (delta)**2
    weighted_delta_sq = (1/weighting.exp())*delta_sq + weighting
    return torch.mean(weighted_delta_sq), torch.mean(delta_sq)

def log_lv_loss(pred, target, weighting, stop_gradient=False, eps=0.01):
    """Computes the mean squared L2 loss."""
    if stop_gradient:
        actual_target = torch.detach(target)
    else:
        actual_target = target
    delta = pred - actual_target
    err = (delta)**2
    mse_loss = torch.sum(err, dim=list(range(1, len(pred.shape))))
    mean_loss = torch.mean(err, dim=list(range(1, len(pred.shape))))
    # Reshape mean_loss to match weighting for broadcasting: (B,) -> (B, 1, 1, 1)
    mean_loss = broadcast_to_shape(mean_loss, weighting.shape)
    log_loss = torch.log((1 / weighting.exp()) * mean_loss + eps) + weighting
    return torch.mean(log_loss), torch.mean(mse_loss)

def adaptive_loss(pred, target, weighting, p, c, stop_gradient=False):
    """Computes the adaptively weighted squared L2 loss.
    Loss = w * ||pred - target||^2, where w = 1 / (||pred - target||^2 + c)^p.
    """
    if stop_gradient:
        actual_target = torch.detach(target)
    else:
        actual_target = target

    delta_sq = (pred - actual_target)**2
    # norm based weighting
    delta_sq_norm = torch.mean(delta_sq, dim=tuple(range(1, len(delta_sq.shape))))
    weight = 1.0 / (delta_sq_norm + c) ** p
    weight = torch.detach(weight)  
    weight = broadcast_to_shape(weight, delta_sq.shape)
    delta_sq = delta_sq * weight
    weighted_delta_sq = delta_sq / weighting.exp() + weighting
    return torch.mean(delta_sq), torch.mean(weighted_delta_sq)
