import torch
from distcfm.losses.utils import l2_loss, adaptive_loss, log_lv_loss

def sample_t_cond(N, step, cfg):
    """Samples conditioning time t_cond with an annealing schedule."""
    n_warmup_steps = cfg.trainer.t_cond_warmup_steps
    t_max = cfg.SI.t_max

    if step < n_warmup_steps:
        # Phase 1: Learning the standard flow map.
        return torch.zeros(N)
    else:
        # Phase 2: Train on the full range of noise levels.
        probs = torch.full((N,), 1.0 - cfg.trainer.t_cond_0_rate)
        return torch.rand(N) ** cfg.trainer.t_cond_power * t_max * torch.bernoulli(probs)
    
def sample_s_u(N, step, cfg):
    n_warmup_steps = cfg.trainer.num_warmup_steps
    anneal_end_step = cfg.trainer.anneal_end_step
    
    t_batch = torch.rand(N, 2) * cfg.SI.t_max  # Shape: [B, 2]
    t1, t2 = t_batch[:, 0], t_batch[:, 1]

    t_min = torch.min(t1, t2)
    t_max = torch.max(t1, t2)

    mid = (t_min + t_max) / 2
    dist = t_max - t_min

    def warmup_phase(): 
        # Phase 1: Learning the diagonal.
        return t1, t1
    def anneal_phase(): 
        # Phase 2: Expanding the jump.
        anneal_duration = anneal_end_step - n_warmup_steps
        progress = (step - n_warmup_steps) / max(anneal_duration, 1)
        max_step_size = torch.clamp(torch.tensor(progress), min=0, max=1)
        s = mid - max_step_size * dist / 2
        t = mid + max_step_size * dist / 2
        return s, t
    def final_phase(): 
        # Phase 3: All jump sizes.
        return t_min, t_max
    
    if step < n_warmup_steps:
        return warmup_phase()
    elif step < anneal_end_step:
        return anneal_phase()
    else:
        return final_phase()

def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

def get_consistency_loss_fn(cfg, SI):
    def loss_fn(model, weighting_model, x1, labels, step, ema_state=None, teacher_model=None):
        # --- 1. Generate Conditioning Variables ---
        device = x1.device
        N = x1.shape[0]  # batch size
        t_cond = sample_t_cond(N, step, cfg) # Shape: [B,]
        t_cond = t_cond.to(device)

        alpha_t_cond, beta_t_cond = SI.get_coefficients(t_cond) # Shape: [B,]
        alpha_t_cond, beta_t_cond = broadcast_to_shape(alpha_t_cond, x1.shape), broadcast_to_shape(beta_t_cond, x1.shape)

        noise_cond = torch.randn_like(x1, device=x1.device)
        xt_cond = alpha_t_cond * noise_cond + beta_t_cond * x1

        # --- 2. Flow Matching Loss (on the diagonal) ---
        x0 = torch.randn_like(x1)

        s_uniform = torch.rand(N,)
        s_uniform = s_uniform.to(device)
        expanded_s_uniform = broadcast_to_shape(s_uniform, x1.shape)
        Is = (1 - expanded_s_uniform) * x0 + expanded_s_uniform * x1

        if cfg.loss.distill_fm:
            eps = 1e-4
            posterior_var = ((1-t_cond)**2 * (1-s_uniform)**2) / (t_cond**2 * (1-s_uniform)**2 + (1-t_cond)**2 * s_uniform**2).clamp_min(eps)
            t_star = 1 / (1 + torch.sqrt(posterior_var))
            t_star_expanded = broadcast_to_shape(t_star, x1.shape)
            posterior_var_expanded = broadcast_to_shape(posterior_var, x1.shape)
            t_cond_expanded = broadcast_to_shape(t_cond, x1.shape)
            s_uniform_expanded = broadcast_to_shape(s_uniform, x1.shape)
            x_star = t_star_expanded *  posterior_var_expanded * (t_cond_expanded / (1-t_cond_expanded).clamp_min(eps)**2 * xt_cond + s_uniform_expanded / (1-s_uniform_expanded).clamp_min(eps)**2 * Is)
            # print(t_star.shape, x_star.shape)
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=True, dtype=x1.dtype if x1.dtype in [torch.bfloat16, torch.float16] else None):
                    v_star = teacher_model.v(t_star, t_star, x_star, torch.zeros_like(t_star), torch.zeros_like(x_star), class_labels=labels)
                    post_mean = x_star + (1.0 - t_star_expanded) * v_star
                    dIsds = (post_mean - Is) / (1.0 - s_uniform_expanded).clamp_min(eps)
        else:
            dIsds = x1 - x0

        fm_pred = model.v(s_uniform, s_uniform, Is, t_cond, xt_cond, class_labels=labels)
        if cfg.model.learn_loss_weighting:
            fm_loss_weighting = weighting_model(s_uniform, s_uniform, t_cond, ema_state=ema_state)
        else:
            fm_loss_weighting = torch.zeros_like(fm_pred)

        if cfg.loss.fm_loss_type == "l2":
            fm_loss, fm_loss_unweighted = l2_loss(fm_pred, dIsds,
                                                  fm_loss_weighting)
        elif cfg.loss.fm_loss_type == "lv":
            fm_loss, fm_loss_unweighted = log_lv_loss(fm_pred, dIsds,
                                                      fm_loss_weighting)
        elif cfg.loss.fm_loss_type == "adaptive":
            fm_loss, fm_loss_unweighted = adaptive_loss(fm_pred, dIsds, fm_loss_weighting, 
                                                        cfg.loss.fm_adaptive_loss_p,
                                                        cfg.loss.fm_adaptive_loss_c)
        else:
            raise ValueError(f"Unknown flow matching loss type: {cfg.loss.fm_loss_type}")

        s, u = sample_s_u(N, step, cfg)  # [B,]
        s, u = s.to(device), u.to(device)
        
        # --- 3. Distillation Loss (on the off-diagonal s<u) ---
        if step > cfg.trainer.num_warmup_steps:
            x0 = torch.randn_like(x1)
            expanded_s = broadcast_to_shape(s, x1.shape)
            Is = (1 - expanded_s) * x0 + expanded_s * x1

            if cfg.loss.distillation_type == "lsd":
                Xsu_fn = lambda s, u, x: model(s, u, x, t_cond, xt_cond, class_labels=labels)
                primals = (s, u, Is)
                tangents = (torch.zeros_like(s, device=device), torch.ones_like(u, device=device), torch.zeros_like(Is, device=device))
                Xsu, dXdu = torch.func.jvp(Xsu_fn, primals, tangents)
                vuu = model.v(u, u, Xsu, t_cond, xt_cond, class_labels=labels)
                
                distillation_student = vuu
                distillation_teacher = dXdu

            elif cfg.loss.distillation_type == "mf":
                vsu_fn = lambda s, u, x: model.v(s, u, x, t_cond, xt_cond, class_labels=labels)
                vss = vsu_fn(s, s, Is)
                primals = (s, u, Is)
                tangents = (torch.ones_like(s, device=device), torch.zeros_like(u, device=device), vss)
                vsu, jvp = torch.func.jvp(vsu_fn, primals, tangents)
                
                distillation_student = vsu
                distillation_teacher = vss + broadcast_to_shape(u-s, jvp.shape) * jvp
            elif cfg.loss.distillation_type == "psd": 
                vsu_fn = lambda s, u, x: model.v(s, u, x, t_cond, xt_cond, class_labels=labels)
                gamma = torch.rand_like(s, device=device)
                w = s + gamma * (u - s)
                vsw = model.v(s, w, Is, t_cond, xt_cond, class_labels=labels)
                Xsw = model.X(s, w, Is, vsw)

                distillation_student = vsu_fn(s, u, Is)
                expanded_gamma = broadcast_to_shape(gamma, x1.shape)
                distillation_teacher = expanded_gamma * vsw + (1-expanded_gamma) * vsu_fn(w, u, Xsw)
            else:
                raise ValueError(f"Unknown distillation loss type: {cfg.loss.distillation_type}")
            
            if cfg.model.learn_loss_weighting:
                distill_loss_weighting = weighting_model(s, u, t_cond, ema_state=ema_state)
            else:
                distill_loss_weighting = torch.zeros_like(distillation_student)

            if cfg.loss.distillation_loss_type == "l2":
                distillation_loss, distillation_loss_unweighted = l2_loss(distillation_student, distillation_teacher,
                                                                        distill_loss_weighting,
                                                                        cfg.loss.distill_teacher_stop_grad,)
            elif cfg.loss.distillation_loss_type == "lv":
                distillation_loss, distillation_loss_unweighted = log_lv_loss(distillation_student, distillation_teacher,
                                                                        distill_loss_weighting,
                                                                        cfg.loss.distill_teacher_stop_grad,)
            elif cfg.loss.distillation_loss_type == "adaptive":
                distillation_loss, distillation_loss_unweighted = adaptive_loss(distillation_student, distillation_teacher,
                                                                                distill_loss_weighting, cfg.loss.distill_adaptive_loss_p, 
                                                                                cfg.loss.distill_adaptive_loss_c,
                                                                                cfg.loss.distill_teacher_stop_grad,)
            else:
                raise ValueError(f"Unknown distillation loss type: {cfg.loss.distillation_loss_type}")
        else:
            distillation_loss = torch.tensor(0.0, device=device)
            distillation_loss_unweighted = torch.tensor(0.0, device=device)

        return {"fm_loss": fm_loss, "distillation_loss": distillation_loss}, {"fm_loss_unweighted": fm_loss_unweighted, 
                                                                                                    "distillation_loss_unweighted": distillation_loss_unweighted}


    return loss_fn
