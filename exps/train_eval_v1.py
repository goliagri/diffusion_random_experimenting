import numpy as np
import os
import torch as th
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist

from guided_diffusion.script_util import create_gaussian_diffusion
from guided_diffusion.resample import UniformSampler


def train_model(dataset, model, par):
    """
    Train a diffusion model with an arbitrary denoising network (MLP, UNet, GNN, etc.).
    
    The model must accept: model(x, t, **kwargs) where:
        - x: noisy input tensor [N, *input_shape]
        - t: timestep tensor [N]
        - **kwargs: optional conditioning info
    
    Args:
        dataset: A torch Dataset that yields (x, cond_dict) tuples where:
                 - x is the data tensor [*input_shape]
                 - cond_dict is a dict of conditioning tensors (can be empty {})
        model: The denoising model
        par: Dict with training parameters
    """
    required_keys = ['batch_size', 'learning_rate', 'num_epochs', 'diffusion_steps', 'ema_rate']
    for key in required_keys:
        if key not in par:
            raise ValueError("Missing required par variable: {}".format(key))
    
    batch_size = par['batch_size']
    learning_rate = par['learning_rate']
    num_epochs = par['num_epochs']
    device = par['device'] if 'device' in par else ('cuda' if th.cuda.is_available() else 'cpu')
    diffusion_steps = par['diffusion_steps']
    ema_rate = par['ema_rate']
    output_dir = par.get('output_dir', './checkpoints')
    save_ckpts = par.get('save_ckpts', False)
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize diffusion
    predict_xstart = par.get('predict_xstart', True)
    diffusion = create_gaussian_diffusion(steps=diffusion_steps, predict_xstart=predict_xstart)
    schedule_sampler = UniformSampler(diffusion)
    
    model.train()
    
    # EMA parameters
    ema_params = [p.clone().detach() for p in model.parameters()]
    
    # Calculate total steps for single progress bar
    total_batches = len(dataloader) * num_epochs
    
    global_step = 0
    pbar = tqdm(total=total_batches, desc="Training")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_data in dataloader:
            # Handle different dataset return formats
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                x_start, cond = batch_data
                if not isinstance(cond, dict):
                    cond = {'y': cond}  # Wrap non-dict conditions
            else:
                x_start = batch_data
                cond = {}
            
            x_start = x_start.to(device)
            cond = {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in cond.items()}
            
            # Sample timesteps
            t, weights = schedule_sampler.sample(x_start.shape[0], device)
            
            # Compute training loss
            losses = diffusion.training_losses(model, x_start, t, model_kwargs=cond)
            loss = (losses["loss"] * weights).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA
            with th.no_grad():
                for ema_p, p in zip(ema_params, model.parameters()):
                    ema_p.mul_(ema_rate).add_(p, alpha=1 - ema_rate)
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(epoch=epoch+1, loss=f"{loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches
        
        # Only print every 10 epochs
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.6f}")
        
        # Save checkpoint each epoch (but silently)
        if save_ckpts:
            checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
            th.save({
                'model_state_dict': model.state_dict(),
                'ema_params': ema_params,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
            }, checkpoint_path)
    
    pbar.close()
    return model


def normalize_range(x, low=-1, high=1):
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = ((high - low) * x) + low
    return x


def sample_from_model(model, diffusion, shape, device, num_samples=1, 
                      cond=None, use_ddim=True, ddim_steps=50, eta=0.0,
                      clip_denoised=False, progress=True, predict_xstart=True):
    """
    Generate samples from a trained diffusion model.
    
    Args:
        model: The trained denoising model
        diffusion: The diffusion object
        shape: Shape of a single sample (without batch dim), e.g. (channels, height, width) or (feature_dim,)
        device: Device to generate on
        num_samples: Number of samples to generate
        cond: Optional conditioning dict
        use_ddim: Whether to use DDIM sampling (faster)
        ddim_steps: Number of DDIM steps if using DDIM
        eta: DDIM eta parameter (0 = deterministic)
        clip_denoised: Whether to clip outputs to [-1, 1]. Set False for regression tasks.
        progress: Show progress bar
        predict_xstart: Whether model predicts x_start (True) or noise (False)
        
    Returns:
        Tensor of samples [num_samples, *shape]
    """
    model.eval()
    full_shape = (num_samples, *shape)
    
    # Expand conditioning tensors to match num_samples if needed
    model_kwargs = {}
    if cond is not None:
        for k, v in cond.items():
            if isinstance(v, th.Tensor) and v.shape[0] == 1 and num_samples > 1:
                # Expand batch dim to match num_samples
                model_kwargs[k] = v.expand(num_samples, *v.shape[1:])
            else:
                model_kwargs[k] = v
    
    with th.no_grad():
        if use_ddim:
            # Create a respaced diffusion for DDIM
            ddim_diffusion = create_gaussian_diffusion(
                steps=diffusion.num_timesteps,
                predict_xstart=predict_xstart,
                timestep_respacing=f"ddim{ddim_steps}"
            )
            samples = ddim_diffusion.ddim_sample_loop(
                model,
                full_shape,
                model_kwargs=model_kwargs,
                device=device,
                eta=eta,
                clip_denoised=clip_denoised,
                progress=progress
            )
        else:
            samples = diffusion.p_sample_loop(
                model,
                full_shape,
                model_kwargs=model_kwargs,
                device=device,
                clip_denoised=clip_denoised,
                progress=progress
            )
    
    model.train()
    return samples


def eval_model(dataset, model, par):
    """
    Evaluate a diffusion model by computing reconstruction loss.
    
    Args:
        dataset: Evaluation dataset yielding (x, cond) tuples
        model: The trained model
        par: Parameter dict with 'diffusion_steps', 'device'
        
    Returns:
        Average MSE loss over the dataset
    """
    required_keys = ['diffusion_steps']
    for key in required_keys:
        if key not in par:
            raise ValueError("Missing required par variable: {}".format(key))
    
    diffusion_steps = par['diffusion_steps']
    device = par['device'] if 'device' in par else ('cuda' if th.cuda.is_available() else 'cpu')
    num_samples_per_point = par.get('num_samples_per_point', 1)
    use_ddim = par.get('use_ddim', True)
    ddim_steps = par.get('ddim_steps', 100)
    predict_xstart = par.get('predict_xstart', True)
    
    diffusion = create_gaussian_diffusion(steps=diffusion_steps, predict_xstart=predict_xstart)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    criterion = th.nn.MSELoss()
    
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    all_preds = []
    all_targets = []
    all_pred_vars = []
    
    for batch_data in tqdm(dataloader, desc="Evaluating"):
        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
            x_target, cond = batch_data
            if not isinstance(cond, dict):
                cond = {'y': cond}
        else:
            x_target = batch_data
            cond = {}
        
        x_target = x_target.to(device)
        cond = {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in cond.items()}
        
        # Generate multiple samples for uncertainty estimation
        sample_shape = x_target.shape[1:]  # Remove batch dim
        
        with th.no_grad():
            # Sample from model
            samples = sample_from_model(
                model, diffusion, sample_shape, device,
                num_samples=num_samples_per_point,
                cond=cond if cond else None,
                use_ddim=use_ddim,
                ddim_steps=ddim_steps,
                progress=False,
                predict_xstart=predict_xstart
            )
            
            # Compute mean prediction and variance
            pred_mean = samples.mean(dim=0, keepdim=True)
            pred_var = samples.var(dim=0, keepdim=True) if num_samples_per_point > 1 else th.zeros_like(pred_mean)
            
            loss = criterion(pred_mean, x_target)
            total_loss += loss.item()
            num_samples += 1
            
            all_preds.append(pred_mean.cpu())
            all_targets.append(x_target.cpu())
            all_pred_vars.append(pred_var.cpu())
    
    avg_loss = total_loss / num_samples
    print("Evaluation - Average MSE: {:.6f}".format(avg_loss))
    
    model.train()
    
    return {
        'avg_loss': avg_loss,
        'predictions': th.cat(all_preds, dim=0),
        'targets': th.cat(all_targets, dim=0),
        'pred_vars': th.cat(all_pred_vars, dim=0),
    }


def viz_1d_regression(train_dataset, eval_dataset, model, par):
    """
    Visualize 1D regression results with uncertainty.
    Assumes data is 1D: x -> y mapping.
    
    Args:
        train_dataset: Training dataset for reference
        eval_dataset: Evaluation dataset
        model: Trained model
        par: Parameters including 'output_dir', 'diffusion_steps'
    """
    required_keys = ['diffusion_steps', 'output_dir']
    for key in required_keys:
        if key not in par:
            raise ValueError("Missing required par variable: {}".format(key))
    
    output_dir = par['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Get eval results
    eval_results = eval_model(eval_dataset, model, par)
    
    preds = eval_results['predictions'].squeeze()
    targets = eval_results['targets'].squeeze()
    pred_vars = eval_results['pred_vars'].squeeze()
    
    # Normalize variance for visualization
    au = normalize_range(pred_vars, low=0.0, high=1.0) if pred_vars.sum() > 0 else pred_vars
    
    # Extract x and y values from datasets
    # Dataset returns (y, cond_dict) where cond_dict = {'cond': x}
    train_xs = []
    train_ys = []
    eval_xs = []
    
    for d in train_dataset:
        if isinstance(d, (list, tuple)) and len(d) == 2:
            y_val, cond = d
            train_ys.append(y_val)
            # Extract x from conditioning dict
            if isinstance(cond, dict) and 'cond' in cond:
                train_xs.append(cond['cond'])
            else:
                train_xs.append(cond)
        else:
            train_ys.append(d)
            train_xs.append(len(train_xs))  # Fallback to index
    
    for d in eval_dataset:
        if isinstance(d, (list, tuple)) and len(d) == 2:
            _, cond = d
            if isinstance(cond, dict) and 'cond' in cond:
                eval_xs.append(cond['cond'])
            else:
                eval_xs.append(cond)
        else:
            eval_xs.append(len(eval_xs))  # Fallback to index
    
    train_xs = np.array([x.numpy() if hasattr(x, 'numpy') else x for x in train_xs]).flatten()
    train_ys = np.array([y.numpy() if hasattr(y, 'numpy') else y for y in train_ys]).flatten()
    eval_xs = np.array([x.numpy() if hasattr(x, 'numpy') else x for x in eval_xs]).flatten()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(train_xs, train_ys, s=5, c="gray", label="Train Data", alpha=0.5)
    plt.scatter(eval_xs, preds.numpy(), c='blue', s=10, label="Prediction")
    
    if au.sum() > 0:
        plt.fill_between(
            eval_xs,
            preds.numpy() - au.numpy(),
            preds.numpy() + au.numpy(),
            color='lightsalmon',
            alpha=0.4,
            label="Uncertainty"
        )
    
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Diffusion Model Predictions")
    plt.savefig(os.path.join(output_dir, "predictions.pdf"))
    plt.close()
    
    print(f"Visualization saved to {os.path.join(output_dir, 'predictions.pdf')}")
