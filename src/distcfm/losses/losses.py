import torch
from distcfm.losses.utils import l2_loss, adaptive_loss, log_lv_loss, compute_loss

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
    def loss_fn(model, weighting_model, x1, labels, step, ema_state=None, teacher_model=None, null_labels=None):
        if cfg.loss.model_guidance and cfg.loss.distillation_type != "mf":
            raise ValueError("Model guidance is only supported with MF distillation.")

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

        # Standard FM target
        dIsds = x1 - x0
        
        # Model prediction
        if cfg.loss.model_guidance:
            fm_pred = model.v(s_uniform, s_uniform, Is, t_cond, xt_cond, class_labels=labels,
                              cfg_scale=torch.ones_like(s_uniform, device=device))
        else:
            fm_pred = model.v(s_uniform, s_uniform, Is, t_cond, xt_cond, class_labels=labels)
        
        if cfg.model.learn_loss_weighting:
            fm_loss_weighting = weighting_model(s_uniform, t_cond)
        else:
            fm_loss_weighting = torch.zeros_like(fm_pred)

        # Standard FM Loss
        fm_loss, fm_loss_unweighted = compute_loss(
            fm_pred, dIsds, fm_loss_weighting, 
            cfg.loss.fm_loss_type,
            adaptive_p=cfg.loss.get("fm_adaptive_loss_p"),
            adaptive_c=cfg.loss.get("fm_adaptive_loss_c")
        )
        
        # for logging 
        with torch.no_grad():
            fm_loss_l2 = (fm_pred - dIsds)**2
            fm_loss_l2 = fm_loss_l2.mean()

        # Learn the amortized velocity field (w != 1) 
        if cfg.loss.model_guidance:
            ws = cfg.model.model_guidance_class_ws 
            rand_indices = torch.randint(0, len(ws), (N,))
            cfg_scale = torch.tensor([ws[i] for i in rand_indices], device=device)
            v_cfg_pred = model.v(s_uniform, s_uniform, Is, t_cond, xt_cond, class_labels=labels,    
                                 cfg_scale=cfg_scale)
            
            with torch.no_grad(): 
                v_cfg_target = model.v_cfg(s_uniform, s_uniform, Is, t_cond, xt_cond, 
                                          class_labels=labels, cfg_scales=cfg_scale)
            
            model_guidance_loss, model_guidance_loss_unweighted = compute_loss(
                        v_cfg_pred, v_cfg_target, fm_loss_weighting, 
                        cfg.loss.fm_loss_type,
                        adaptive_p=cfg.loss.get("fm_adaptive_loss_p"),
                        adaptive_c=cfg.loss.get("fm_adaptive_loss_c"))
        
        # Distilled FM Loss
        distill_fm_loss = torch.tensor(0.0, device=device)
        distill_fm_loss_unweighted = torch.tensor(0.0, device=device)

        if cfg.loss.distill_fm:
            eps = 1e-6
            
            s = expanded_s_uniform
            t = broadcast_to_shape(t_cond, x1.shape)
            one_minus_s = 1 - s
            
            denom = (t**2 * one_minus_s**2 + (1-t)**2 * s**2).clamp_min(eps)                                                                                                                                                            
            P_norm = (1-t)**2 / denom
            sqrt_P_norm = torch.sqrt(P_norm)
            
            t_star = 1 / (1 + one_minus_s * sqrt_P_norm)
            
            coeff_cond = t_star * one_minus_s**2 * t / denom
            coeff_Is = t_star * s * P_norm
            x_star = coeff_cond * xt_cond + coeff_Is * Is
            
            with torch.no_grad():
                v_star = teacher_model.v(t_star.view(N), t_star.view(N), x_star, torch.zeros(N, device=device), torch.zeros_like(x_star), class_labels=labels)
                
                term2 = t_star * sqrt_P_norm * v_star
                
                diff_div_x = ((1-t)**2 * (1 + s) - t**2 * one_minus_s) / denom
                B_minus_1_div_x = (diff_div_x - (P_norm + sqrt_P_norm)) / (1 + one_minus_s * sqrt_P_norm)
                A_div_x = t_star * one_minus_s * t / denom
                
                term1 = A_div_x * xt_cond + B_minus_1_div_x * Is
                dIsds_distill = term1 + term2
            
            # Calculate loss
            distill_loss_weighting = torch.zeros_like(fm_pred)
            distill_fm_loss, distill_fm_loss_unweighted = compute_loss(
                fm_pred, dIsds_distill, distill_loss_weighting,
                cfg.loss.distill_fm_loss_type,
                adaptive_p=cfg.loss.get("fm_adaptive_loss_p"),
                adaptive_c=cfg.loss.get("fm_adaptive_loss_c")
            )
            # print(f"time_diff: {(t_star - s_uniform).mean().item():.4f}, x_diff: {(x_star-Is).mean().item():.4f}, v_diff: {(v_star-dIsds_distill).mean().item():.4f},  distill_fm_loss: {distill_fm_loss.item():.6f}")
        
        with torch.no_grad():
            distill_fm_loss_l2 = (fm_pred - dIsds_distill)**2
            distill_fm_loss_l2 = distill_fm_loss_l2.mean()

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
                if not cfg.loss.model_guidance: # standard 
                    vsu_fn = lambda s, u, x: model.v(s, u, x, t_cond, xt_cond, class_labels=labels)
                    vss = vsu_fn(s, s, Is)
                else:
                    # get the guidance scales for distillation
                    with torch.no_grad():
                        scales = torch.randint(0, len(cfg.model.model_guidance_class_ws), (N,))
                        cfg_scale = torch.tensor([cfg.model.model_guidance_class_ws[i] for i in cfg_scale], device=device)
                        p = cfg.loss.model_guidance_distill_base_prob
                        cfg_mask = torch.bernoulli(torch.full(N, p, device=self.device)).bool()
                        cfg_scales_distill = torch.where(cfg_mask, torch.ones_like(cfg_scale, device=device), cfg_scale)
                        vss = model.v_cfg(s, s, Is, t_cond, xt_cond, class_labels=labels, cfg_scales=cfg_scales_distill)
                    
                    vsu_fn = lambda s, u, x: model.v(s, u, x, t_cond, xt_cond, class_labels=labels, cfg_scale=cfg_scales_distill)
                
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
            
            distill_loss_weighting = torch.zeros_like(distillation_student)

            distillation_loss, distillation_loss_unweighted = compute_loss(
                distillation_student, distillation_teacher, distill_loss_weighting,
                cfg.loss.distillation_loss_type,
                adaptive_p=cfg.loss.get("distill_adaptive_loss_p"),
                adaptive_c=cfg.loss.get("distill_adaptive_loss_c"),
                stop_gradient=cfg.loss.distill_teacher_stop_grad
            )
        else:
            distillation_loss = torch.tensor(0.0, device=device)
            distillation_loss_unweighted = torch.tensor(0.0, device=device)
        
        with torch.no_grad():
            distillation_loss_l2 = (distillation_student - distillation_teacher)**2
            distillation_loss_l2 = distillation_loss_l2.mean()

        return {"fm_loss": fm_loss, "distill_fm_loss": distill_fm_loss, "distillation_loss": distillation_loss, "model_guidance_loss": model_guidance_loss}, {"fm_loss_unweighted": fm_loss_unweighted, 
                                                                                                                                                              "distill_fm_loss_unweighted": distill_fm_loss_unweighted,
                                                                                                                                                              "distillation_loss_unweighted": distillation_loss_unweighted, 
                                                                                                                                                              "model_guidance_loss_unweighted": model_guidance_loss_unweighted,
                                                                                                                                                              "fm_loss_l2": fm_loss_l2,
                                                                                                                                                              "distill_fm_loss_l2": distill_fm_loss_l2,
                                                                                                                                                              "distillation_loss_l2": distillation_loss_l2}


    return loss_fn
