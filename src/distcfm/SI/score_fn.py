import torch

def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

def get_unconditional_score_fn(model, sde):
    def score_fn(x, t):
        alpha_t, var_t = sde.get_coefficients(t)
        var_t = broadcast_to_shape(var_t, x.shape)
        eps_pred = model(x, t)
        score = - eps_pred / torch.sqrt(var_t)
        score = score.detach()

        return score
    return score_fn

def get_dps_score_fn(model, sde, log_likelihood_fn, inverse_scaler):
    def score_fn(x, t, y):
        alpha_t, var_t = sde.get_coefficients(t)
        alpha_t, var_t = broadcast_to_shape(alpha_t, x.shape), broadcast_to_shape(var_t, x.shape)
        eps_pred = model(x, t)
        x0_hat = (x - eps_pred*torch.sqrt(var_t))/alpha_t 
        log_liks = log_likelihood_fn(inverse_scaler(x0_hat), y)
        log_liks = log_liks.sum()
        grad = torch.autograd.grad(log_liks, x)[0]
        return grad
    return score_fn

def get_iwae_score_fn(distributional_model, log_likelihood_fn, inverse_scaler, mc_samples=10,
                      type="cfm", sde=None):
    if type == "cfm":
        def score_fn(x, t, y, return_ess = False): # this assumes t from the diffusion process and t_cond are aligned in noise level
            B = x.shape[0]
            x_batched = x.repeat_interleave(mc_samples, dim=0)  # [N*B, C, H, W]
            t_batched = t.repeat_interleave(mc_samples, dim=0)  # [N*B]
            y_batched = y.repeat_interleave(mc_samples, dim=0)  # [N*B, C, H, W]
            
            x0_flow = torch.randn_like(x_batched) # [N*B, C, H, W]
            t0_flow = torch.zeros_like(t_batched)  # [N*B]
            t1_flow = torch.ones_like(t_batched)   # [N*B]

            vst = distributional_model.v(t0_flow, t1_flow, x0_flow, 
                                         t_batched, x_batched)
            x0_samples = distributional_model.X(t0_flow, t1_flow, x0_flow, vst) # [N*B, C, H, W]
            log_liks = log_likelihood_fn(inverse_scaler(x0_samples), y_batched) # [N*B]
            log_liks = log_liks.view(B, mc_samples) # [B, N]
            iwae = torch.logsumexp(log_liks, dim=1) - torch.log(torch.tensor(mc_samples, dtype=log_liks.dtype, device=log_liks.device))
            iwae = iwae.sum()  # [B,]
            grad = torch.autograd.grad(iwae, x)[0]
            
            # exploratory analysis: ess, grad norm
            log_w = log_liks - torch.logsumexp(log_liks, dim=1, keepdim=True)     # [B, N]
            w = torch.exp(log_w)                                                  # [B, N], rows sum to 1
            ess = 1.0 / torch.sum(w ** 2, dim=1)                                  # [B], ESS in [1, N]
            ess = ess.mean()  # average over batch - 1 to N
            ess_rel = ess / mc_samples # relative ESS in [1/N, 1]
            grad_norm = torch.norm(grad.view(B, -1), dim=1)
            
            if not return_ess:
                return grad
            return grad, ess, ess_rel, log_liks, grad_norm
    
    elif type == "cfm_combined":
        assert sde is not None, "sde must be provided for cfm_combined type"
        def score_fn(x, t, y): 
            B = x.shape[0]
            x_batched = x.repeat_interleave(mc_samples, dim=0)  # [N*B, C, H, W]
            t_batched = t.repeat_interleave(mc_samples, dim=0)  # [N*B]
            y_batched = y.repeat_interleave(mc_samples, dim=0)  # [N*B, C, H, W]
            
            x0_flow = torch.randn_like(x_batched) # [N*B, C, H, W]
            t0_flow = torch.zeros_like(t_batched)  # [N*B]
            t1_flow = torch.ones_like(t_batched)   # [N*B]

            vst = distributional_model.v(t0_flow, t1_flow, x0_flow, 
                                         t_batched, x_batched)
            x0_samples = distributional_model.X(t0_flow, t1_flow, x0_flow, vst) # [N*B, C, H, W]
            log_liks = log_likelihood_fn(inverse_scaler(x0_samples), y_batched) # [N*B]
            log_liks = log_liks.view(B, mc_samples) # [B, N]
            iwae = torch.logsumexp(log_liks, dim=1) - torch.log(torch.tensor(mc_samples, dtype=log_liks.dtype, device=log_liks.device))
            iwae = iwae.sum()  # [B,]
            grad = torch.autograd.grad(iwae, x)[0]
            x0_samples = x0_samples.view(B, mc_samples, *x0_samples.shape[1:]) # [B, N, C, H, W]
            x0_mean = x0_samples.mean(dim=1) # [B, C, H, W]
            alpha_t, var_t = sde.get_coefficients(t) # get coefficients
            alpha_t = broadcast_to_shape(alpha_t, x0_mean.shape)
            var_t = broadcast_to_shape(var_t, x0_mean.shape)
            uncond_score = - (x - alpha_t * x0_mean) / var_t
            grad = grad + uncond_score
            return grad
    elif type == "cfm_approximate":
        def score_fn(x, t, y): 
            B = x.shape[0]
            x_batched = x.repeat_interleave(mc_samples, dim=0)  # [N*B, C, H, W]
            t_batched = t.repeat_interleave(mc_samples, dim=0)  # [N*B]
            y_batched = y.repeat_interleave(mc_samples, dim=0)  # [N*B, C, H, W]
            
            x0_flow = torch.randn_like(x_batched) # [N*B, C, H, W]
            t0_flow = torch.zeros_like(t_batched)  # [N*B]
            t1_flow = torch.ones_like(t_batched)   # [N*B]
            
            # [N*B, C, H, W]
            with torch.no_grad():
                vst = distributional_model.v(t0_flow, t1_flow, x0_flow, 
                                            t_batched, x_batched.detach())
                x0_samples = distributional_model.X(t0_flow, t1_flow, x0_flow, vst) 
            
            # straight-through gradient estimator
            x0_samples = x0_samples.detach() + x_batched - x_batched.detach()
            log_liks = log_likelihood_fn(inverse_scaler(x0_samples), y_batched) # [N*B]
            log_liks = log_liks.view(B, mc_samples) # [B, N]
            iwae = torch.logsumexp(log_liks, dim=1) - torch.log(torch.tensor(mc_samples, dtype=log_liks.dtype, device=log_liks.device))
            iwae = iwae.sum()  # [B,]
            grad = torch.autograd.grad(iwae, x)[0]
            return grad
    
    return score_fn

def get_mc_score_fn(distributional_model, log_likelihood_fn, inverse_scaler, sde, mc_samples=10, type="cfm"):
    if type == "cfm":
        @torch.no_grad
        def score_fn(x, t, y): # this assumes t from the diffusion process and t_cond are aligned in noise level
        
            B = x.shape[0]
            x_batched = x.repeat_interleave(mc_samples, dim=0)  # [N*B, C, H, W]
            t_batched = t.repeat_interleave(mc_samples, dim=0)  # [N*B]
            y_batched = y.repeat_interleave(mc_samples, dim=0)  # [N*B, C, H, W]
            
            x0_flow = torch.randn_like(x_batched) # [N*B, C, H, W]
            t0_flow = torch.zeros_like(t_batched)  # [N*B]
            t1_flow = torch.ones_like(t_batched)   # [N*B]
            
            vst = distributional_model.v(t0_flow, t1_flow, x0_flow, 
                                         t_batched, x_batched)
            
            x0_samples = distributional_model.X(t0_flow, t1_flow, x0_flow, vst) # [N*B, C, H, W]
            log_liks = log_likelihood_fn(inverse_scaler(x0_samples), y_batched) # [N*B]
            x0_samples = x0_samples.view(B, mc_samples, *x0_samples.shape[1:]) # [B, N, C, H, W]
            log_liks = log_liks.view(B, mc_samples) # [B, N]
            weights = torch.softmax(log_liks, dim=1).view(B, mc_samples, 1, 1, 1) # [B, N, 1, 1, 1]
            weighted_x0_samples = x0_samples * weights
            
            x0_mean = x0_samples.mean(dim=1)
            x0_wmean = weighted_x0_samples.sum(dim=1)

            alpha_t, var_t = sde.get_coefficients(t)
            alpha_t = broadcast_to_shape(alpha_t, x0_mean.shape)
            var_t = broadcast_to_shape(var_t, x0_mean.shape)
            grad = alpha_t / var_t * (x0_wmean - x0_mean)
            return grad
    else:
        raise NotImplementedError(f"Unimplemented distributional model type: {type}")
    
    return score_fn