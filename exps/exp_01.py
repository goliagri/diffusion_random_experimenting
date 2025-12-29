'''
Experiment 01: Train a diffusion model with an MLP denoiser on a 1D threshold dataset.

Dataset: x ~ N(0,1), y = x+5 if x>0 else x-5
Model: Simple MLP that takes (noisy_y, t, x) and predicts denoised y
'''

import argparse
import torch as th
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'openai_guided-diffusion'))

from dsets.threshold_1d import Threshold1DDataset
from models.mlp_denoiser import MLPDenoiser
import exps.train_eval_v1 as train_eval


def run_experiment(par):
    """Run training and evaluation."""
    
    print("=" * 50)
    print("Experiment: MLP Diffusion on 1D Threshold Dataset")
    print("=" * 50)
    
    # Step 1: Create datasets
    print("\n[1] Creating datasets...")
    train_dataset = Threshold1DDataset(par['dataset_size'], seed=42)
    eval_dataset = Threshold1DDataset(par['eval_size'], seed=123)
    
    print(f"    Train size: {len(train_dataset)}")
    print(f"    Eval size: {len(eval_dataset)}")
    
    # Show some samples
    print("\n    Sample data points:")
    for i in range(3):
        y, cond = train_dataset[i]
        print(f"      x={cond['cond'].item():+.3f} -> y={y.item():+.3f}")
    
    # Step 2: Create model
    print("\n[2] Creating MLP denoiser...")
    model = MLPDenoiser(
        input_dim=1,
        hidden_dim=par['hidden_dim'],
        num_layers=par['num_layers'],
        time_embed_dim=64,
        cond_dim=1
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"    Model parameters: {num_params:,}")
    
    # Step 3: Train
    print("\n[3] Training model...")
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    par['device'] = device
    print(f"    Device: {device}")
    
    model = train_eval.train_model(train_dataset, model, par)
    
    # Step 4: Evaluate
    print("\n[4] Evaluating model...")
    eval_par = par.copy()
    eval_par['num_samples_per_point'] = par.get('num_eval_samples', 10)
    eval_par['use_ddim'] = True
    eval_par['ddim_steps'] = 50
    
    results = train_eval.eval_model(eval_dataset, model, eval_par)
    
    # Step 5: Visualize
    print("\n[5] Generating visualization...")
    train_eval.viz_1d_regression(train_dataset, eval_dataset, model, par)
    
    # Additional visualization: show predicted vs true
    print("\n[6] Sample predictions:")
    preds = results['predictions']
    targets = results['targets']
    
    for i in range(min(5, len(preds))):
        y_true = targets[i].item()
        y_pred = preds[i].item()
        _, cond = eval_dataset[i]
        x_val = cond['cond'].item()
        print(f"    x={x_val:+.3f}: true={y_true:+.3f}, pred={y_pred:+.3f}, err={abs(y_true-y_pred):.3f}")
    
    print("\n" + "=" * 50)
    print("Experiment complete!")
    print(f"Results saved to: {par['output_dir']}")
    print("=" * 50)


def main(par):
    os.makedirs(par['output_dir'], exist_ok=True)
    
    mode = par['mode']
    if mode == 'run_experiment':
        run_experiment(par)
    else:
        print(f"Unknown mode: {mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="1D Threshold Diffusion Experiment")
    
    # Mode
    parser.add_argument('--mode', required=True, choices=['run_experiment'])
    
    # Dataset
    parser.add_argument('--dataset_size', type=int, default=10000, help="Training set size")
    parser.add_argument('--eval_size', type=int, default=200, help="Eval set size")
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=128, help="MLP hidden dimension")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of MLP layers")
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--diffusion_steps', type=int, default=500)
    parser.add_argument('--predict_xstart', action='store_true', default=True, help="Predict x_start (True) or noise (False)")
    parser.add_argument('--no_predict_xstart', action='store_false', dest='predict_xstart', help="Predict noise instead of x_start")
    parser.add_argument('--ema_rate', type=float, default=0.9999)
    parser.add_argument('--save_log_interval', type=int, default=500)
    
    # Evaluation
    parser.add_argument('--num_eval_samples', type=int, default=1, help="Samples per point for uncertainty")
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/exp_01')
    parser.add_argument('--save_ckpts', action='store_true', help="Save model checkpoints each epoch")

    args = parser.parse_args()
    par = vars(args)
    main(par)
