# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path

from fairmotion.models import (
    rnn,
    seq2seq,
    transformer,
    SpatioTemporalTransformer,
    moe
)
from fairmotion.data import amass_dip, bvh
from fairmotion.core import motion as motion_class
from fairmotion.tasks.motion_prediction import generate, metrics, utils
from fairmotion.ops import conversions, motion as motion_ops
from fairmotion.utils import utils as fairmotion_utils

from typing import Any, Callable, Dict, Iterable, List, Tuple
from torch.utils.data import DataLoader
from fairmotion.core.motion import Skeleton


# Set environment variable for MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

LOGGER = logging.getLogger(__name__)

def configure_logging():
    global LOGGER
    logFormatter = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    LOGGER.addHandler(consoleHandler)

def test_single_input(
    model, 
    input_seq, 
    device: str, 
    mean: float, 
    std: float, 
    max_len: int = 1
):
    with torch.no_grad():
        model.eval()
        input_seq = input_seq.to(device)  # Send input to the appropriate device
        pred_seq, dispatch_weights, combine_weights = model(input_seq, tgt=None, max_len=1, teacher_forcing_ratio=0, output_weights=True)
        pred_seq = pred_seq.cpu().numpy()  # Move data to CPU and convert to numpy
        return pred_seq, dispatch_weights.cpu().numpy(), combine_weights.cpu().numpy()

def visualize_weights(weights, title):
    num_plots = weights.shape[-1]  # Get the number of subplots based on the last dimension of the weights tensor

    fig, axs = plt.subplots(1, num_plots, figsize=(15, 5))  # Create subplots
    
    for i in range(num_plots):
        heatmap = torch.tensor(weights[0, :, i, i])  # Convert NumPy array to PyTorch tensor
        axs[i].imshow(heatmap.unsqueeze(0), aspect='auto', cmap='hot')  # Display the heatmap
        axs[i].set_title(f'Subplot {i+1}')  # Set title for the subplot
        axs[i].set_xlabel('Index')  # Set xlabel
        axs[i].set_ylabel('Value')  # Set ylabel
        
    plt.suptitle(title)  # Set main title for all subplots
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")  # Save the figure
    plt.show()

def prepare_model(
    path: Path,
    num_predictions: int,
    hidden_dim: int,
    device: str,
    num_layers: int,
    architecture: str,
    num_heads: int=4,
    src_len: int=120,
    ninp: int=56,
    num_experts: int=16
) -> (
    rnn.RNN |
    seq2seq.Seq2Seq |
    seq2seq.TiedSeq2Seq |
    transformer.TransformerLSTMModel |
    transformer.TransformerModel |
    SpatioTemporalTransformer.TransformerSpatialTemporalModel |
    moe.moe
):
    model = utils.prepare_model(
        input_dim=num_predictions,
        hidden_dim=hidden_dim,
        device=device,
        num_layers=num_layers,
        architecture=architecture,
        num_heads = num_heads,
        src_len=src_len,
        ninp = ninp,
        num_experts=num_experts
    )
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()
    return model

def main(args):
    device = fairmotion_utils.set_device(args.device)
    LOGGER.info("Loading dataset")
    dataset, mean, std = utils.prepare_dataset(
        *[
            args.preprocessed_path.joinpath(f"{split}.pkl")
            for split in ["train", "test", "validation"]
        ],
        batch_size=1,  # Set batch size to 1 to simplify processing
        device=device,
        shuffle=False  # No need to shuffle as we are only taking the first entry
    )

    input_seq =  next(iter(dataset["test"]))[0]

    data_shape = next(iter(dataset["test"]))[0].shape
    num_predictions = data_shape[-1]
    print(data_shape)
    print(num_predictions)

    LOGGER.info("Preparing model")
    model_filename = f"{args.save_model_name}.model"
    model_file_path = args.save_model_path.joinpath(model_filename)
    model = prepare_model(
        model_file_path,
        num_predictions=num_predictions,
        hidden_dim=args.hidden_dim,
        device=device,
        num_layers=args.num_layers,
        architecture=args.architecture,
        num_heads = args.num_heads,
        src_len=120,
        ninp = args.ninp,
        num_experts=args.num_experts
    )
    model.eval()
    pred_seq, dispatch_weights, combine_weights = test_single_input(
        model, input_seq, device, mean, std, args.max_len
    )

    visualize_weights(dispatch_weights, 'Dispatch Weights')
    visualize_weights(combine_weights, 'Combine Weights')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate predictions and post process them"
    )
    parser.add_argument(
        "--preprocessed-path",
        dest="preprocessed_path",
        # type=str,
        type=lambda p: Path(p).expanduser().resolve(strict=True),
        help="Path to folder with pickled files from dataset",
        required=True,
    )
    parser.add_argument(
        "--save-model-path",
        dest="save_model_path",
        # type=str,
        type=lambda p: Path(p).expanduser().resolve(strict=True),
        help="Path to saved models",
        required=True,
    )
    parser.add_argument(
        "--save-model-name",
        dest="save_model_name",
        type=str,
        help="model name",
        required=True,
    )
    parser.add_argument(
        "--save-output-path",
        dest="save_output_path",
        # type=str,
        type=lambda p: Path(p).expanduser().resolve(strict=True),
        help="Path to store predicted motion",
        default=None,
    )
    parser.add_argument(
        "--hidden-dim",
        dest="hidden_dim",
        type=int,
        help="Hidden size of LSTM units in encoder/decoder",
        default=1024,
    )
    parser.add_argument(
        "--ninp",
        type=int,
        help="D dimension joint embedding",
        default=56,
    )
    parser.add_argument(
        "--num-heads",
        dest="num_heads",
        type=int,
        help="Number of heads in each attention block",
        default=4,
    )
    parser.add_argument(
        "--num-layers",
        dest="num_layers",
        type=int,
        help="Number of layers of LSTM/Transformer in encoder/decoder",
        default=1,
    )
    parser.add_argument(
        "--num-experts",
        dest="num_experts",
        type=int,
        help="Number of experts in MOE block",
        default=16,
    )
    parser.add_argument(
        "--max-len",
        type=int,
        help="Length of seq to generate",
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for testing",
        default=64
    )
    parser.add_argument(
        "--shuffle",
        action='store_true',
        help="Use this option to enable shuffling",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Training device",
        default=None,
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Seq2Seq archtiecture to be used",
        default="seq2seq",
        choices=[
            "seq2seq",
            "tied_seq2seq",
            "transformer",
            "transformer_encoder",
            "rnn",
            "STtransformer",
            "moe"
        ],
    )
    args = parser.parse_args()

    # validate provided options
    if not args.preprocessed_path.is_dir():
        raise ValueError(f"Value given for '--prepocessed-path' must be a directory. Given: {args.preprocessed_path}")

    main(args)
