# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt
import re
from multiprocessing import Pool
from pathlib import Path
from collections import OrderedDict
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
    architecture: str,
    device: str, 
    mean: float, 
    std: float, 
    max_len: int = 1
):
    with torch.no_grad():
        model.eval()
        input_seq = input_seq.to(device)  # Send input to the appropriate device
        if architecture == "moe":
            pred_seq, dispatch_weights, combine_weights = model(input_seq, tgt=None, max_len=1, teacher_forcing_ratio=0, output_weights=True)
            return pred_seq.cpu().numpy(), dispatch_weights.cpu().numpy(), combine_weights.cpu().numpy()
        elif architecture == "STtransformer":
            pred_seq, spatial_attention_weights, temporal_attention_weights = model(input_seq, tgt=None, max_len=1, teacher_forcing_ratio=0, output_weights=True)
            return pred_seq.cpu().numpy(), spatial_attention_weights.cpu().numpy(), temporal_attention_weights.cpu().numpy()
        else:
            raise Exception("Unsupported architecture for visualization")

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

def visualize_st_weights(weights, title, ranges, subplot_title):
    fig, axs = plt.subplots(1, len(ranges), figsize=(15, 5))  # Create subplots
    
    for axs_idx, value in enumerate(ranges):
        heatmap = torch.tensor(weights[value-1, :, :])  # Convert NumPy array to PyTorch tensor
        # import pdb;pdb.set_trace()
        axs[axs_idx].imshow(heatmap, aspect='auto', cmap='hot')  # Display the heatmap
        axs[axs_idx].set_title(f'{subplot_title} {value}')  # Set title for the subplot
    
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
    loaded_state_dict = torch.load(path,map_location=torch.device('cpu'))
    formatted_model_state_dict = OrderedDict()
    for key, val in loaded_state_dict["model_state_dict"].items():
        matched_key = re.match("^module.(.*)$", key)
        if matched_key:
            formatted_model_state_dict[matched_key.group(1)] = val
        else:
            formatted_model_state_dict[key] = val
    model.load_state_dict(formatted_model_state_dict)
    model.eval()
    return model


def convert_to_T(
    pred_seqs: np.ndarray,
    src_seqs: np.ndarray,
    tgt_seqs: np.ndarray,
    rep: str
) -> List[np.ndarray]:
    ops = utils.convert_fn_to_R(rep)
    seqs_T = [
        conversions.R2T(utils.apply_ops(seqs, ops))
        for seqs in [pred_seqs, src_seqs, tgt_seqs]
    ]
    return seqs_T


def calculate_metrics(
    pred_seqs: np.ndarray,
    tgt_seqs: np.ndarray
) -> Dict[int, np.floating]:

    metric_frames = [6, 12, 18, 24]
    R_pred, _ = conversions.T2Rp(pred_seqs)
    R_tgt, _ = conversions.T2Rp(tgt_seqs)
    euler_error = metrics.euler_diff(
        R_pred[:, :, amass_dip.SMPL_MAJOR_JOINTS],
        R_tgt[:, :, amass_dip.SMPL_MAJOR_JOINTS],
    )
    euler_error = np.mean(euler_error, axis=0)
    mae = {frame: np.sum(euler_error[:frame]) for frame in metric_frames}
    return mae


def test_model(
    model: (
        rnn.RNN |
        seq2seq.Seq2Seq |
        seq2seq.TiedSeq2Seq |
        transformer.TransformerLSTMModel |
        transformer.TransformerModel |
        SpatioTemporalTransformer.TransformerSpatialTemporalModel |
        moe.moe
    ),
    dataset: Dict[str, DataLoader],
    rep: str,
    device: str,
    mean: float,
    std: float,
    max_len: int=None
) -> Tuple[List[np.ndarray], Dict[int, np.floating]]:
    pred_seqs, src_seqs, tgt_seqs = run_model(
        model, dataset, max_len, device, mean, std,
    )
    seqs_T = convert_to_T(pred_seqs, src_seqs, tgt_seqs, rep)
    # Calculate metric only when generated sequence has same shape as reference
    # target sequence
    if len(pred_seqs) > 0 and pred_seqs[0].shape == tgt_seqs[0].shape:
        mae = calculate_metrics(seqs_T[0], seqs_T[2])

    return seqs_T, mae


def main(args: argparse.Namespace):
    configure_logging(args.architecture, args.save_model_path)
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
    if args.architecture == "moe":
        pred_seq, dispatch_weights, combine_weights = test_single_input(model, input_seq, args.architecture, device, mean, std, args.max_len)
        visualize_weights(dispatch_weights, 'Dispatch Weights')
        visualize_weights(combine_weights, 'Combine Weights')
    elif args.architecture == "STtransformer":
        pred_seq, spatial_attention_weights, temporal_attention_weights = test_single_input(model, input_seq, args.architecture, device, mean, std, args.max_len)
        print(f"{spatial_attention_weights.shape=}")
        print(f"{temporal_attention_weights.shape=}")
        visualize_st_weights(spatial_attention_weights, "Spatial Attention Weights", [30,60,90,120], "Joints at timestep ")
        visualize_st_weights(temporal_attention_weights, "Temporal Attention Weights", [6,12,18,24], "Temporal weights on joints ")

    else:
        raise Exception("Unsupported architecture for visualization")


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
