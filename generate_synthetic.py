import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os

from models import TimeLLM

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic multivariate time series")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the real CSV dataset')
    parser.add_argument('--input_len', type=int, default=128, help='Length of seed window')
    parser.add_argument('--generate_steps', type=int, default=96, help='Number of synthetic steps to generate')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_path', type=str, default='./synthetic_output', help='Where to save the output')
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='Task name')
    parser.add_argument('--enc_in', type=int, default=21, help='Number of encoder input features')
    parser.add_argument('--dec_in', type=int, default=21, help='Number of decoder input features')
    parser.add_argument('--c_out', type=int, default=21, help='Number of output features')
    parser.add_argument('--label_len', type=int, default=0, help='Length of the label sequence')
    parser.add_argument('--pred_len', type=int, default=96, help='Length of prediction')
    parser.add_argument('--d_model', type=int, default=16, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='Dimension of feed-forward network')
    parser.add_argument('--moving_avg', type=int, default=25, help='Window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='Attention factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--embed', type=str, default='timeF', help='Embedding type')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--output_attention', type=bool, default=False, help='Output attention or not')
    parser.add_argument('--patch_len', type=int, default=16, help='Patch length')
    parser.add_argument('--stride', type=int, default=8, help='Stride length')
    parser.add_argument('--prompt_domain', type=int, default=0, help='Whether to use prompt domain info')
    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model type')
    parser.add_argument('--llm_dim', type=int, default=768, help='LLM dimension')
    parser.add_argument('--llm_layers', type=int, default=4, help='Number of LLM layers')
    parser.add_argument('--features', type=str, default='M', help='Features type: S, M, MS')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load data
    df = pd.read_csv(args.data_path)
    df = df.dropna()

    # Drop 'date' column if it's there
    if 'date' in df.columns[0].lower():
      df = df.iloc[:, 1:]
    
    data = df.values

    # Ensure 2D even if single row
    if data.ndim == 1:
      data = np.expand_dims(data, axis=0)

    # Validate feature count
    if data.shape[1] != 21:
      raise ValueError(f"Expected {args.enc_in} features, but got {data.shape[1]}. Check your dataset format.")

    # Normalize
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    data = (data - mean) / std

    seed = data[:args.input_len]
    generated = []

    model_args = args
    model = TimeLLM.Model(model_args).float()
    checkpoint = torch.load(args.model_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    current_input = seed.copy()

    for _ in range(args.generate_steps):
        inp = torch.tensor(current_input[-args.input_len:], dtype=torch.float32).unsqueeze(0)
        inp_mark = torch.zeros_like(inp)  # dummy mark
        dec_inp = torch.zeros((1, 96, 21))  # decoder input (zeroed)
        dec_mark = torch.zeros_like(dec_inp)
        with torch.no_grad():
            output = model(inp, inp_mark, dec_inp, dec_mark)
        next_step = output[:, -1:, :].squeeze(0).cpu().numpy()
        current_input = np.concatenate([current_input, next_step], axis=0)
        generated.append(next_step)

    generated = np.concatenate(generated, axis=0) * std + mean
    seed_real = seed * std + mean

    os.makedirs(args.output_path, exist_ok=True)
    np.save(os.path.join(args.output_path, 'synthetic.npy'), generated)
    np.save(os.path.join(args.output_path, 'seed.npy'), seed_real)

    plt.figure(figsize=(12, 6))
    plt.plot(range(args.input_len), seed_real[:, 0], label='Seed (feature 0)')
    plt.plot(range(args.input_len, args.input_len + args.generate_steps), generated[:, 0], label='Generated (feature 0)')
    plt.legend()
    plt.grid(True)
    plt.title('Synthetic Generation (Feature 0)')
    plt.savefig(os.path.join(args.output_path, 'synthetic_plot.png'))
    plt.show()

if __name__ == '__main__':
    main()
