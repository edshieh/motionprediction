# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import numpy as np
import os
import torch
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

def configure_logging(model_architecture: str, model_save_path: Path):
    global LOGGER
    logFormatter = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGGER.setLevel(logging.INFO)

    log_file = model_save_path.joinpath(f"test_{model_architecture}.log")
    if log_file.exists():
        log_file.unlink()
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(logFormatter)
    LOGGER.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    LOGGER.addHandler(consoleHandler)


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
    state_dict = torch.load(path,map_location=torch.device('cpu'))['model_state_dict']
    if "module." in state_dict.keys()[0]:
        state_dict = {k.partition('module.')[2]: v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_model(
    model: (
        rnn.RNN |
        seq2seq.Seq2Seq |
        seq2seq.TiedSeq2Seq |
        transformer.TransformerLSTMModel |
        transformer.TransformerModel |
        SpatioTemporalTransformer.TransformerSpatialTemporalModel |
        moe.moe
    ),
    data_iter: Dict[str, DataLoader],
    max_len: int,
    device: str,
    mean: float,
    std: float,
    src_len: int
) -> List[np.ndarray]:
    pred_seqs = []
    src_seqs, tgt_seqs = [], []
    for src_seq, tgt_seq in data_iter:
        max_len = max_len if max_len else tgt_seq.shape[1]
        src_seq = src_seq[:, -src_len:, :]
        src_seqs.extend(src_seq.to(device="cpu").numpy())
        tgt_seqs.extend(tgt_seq.to(device="cpu").numpy())
        pred_seq = (
            generate.generate(model, src_seq, max_len, device)
            .to(device="cpu")
            .numpy()
        )
        pred_seqs.extend(pred_seq)
    return [
        utils.unnormalize(np.array(l), mean, std)
        for l in [pred_seqs, src_seqs, tgt_seqs]
    ]


def save_seq(
    i: int,
    pred_seq: np.ndarray,
    src_seq: np.ndarray,
    tgt_seq: np.ndarray,
    skel: Skeleton,
    save_output_path: Path
) -> None:
    # seq_T contains pred, src, tgt data in the same order
    motions = [
        motion_class.Motion.from_matrix(seq, skel)
        for seq in [pred_seq, src_seq, tgt_seq]
    ]
    ref_motion = motion_ops.append(motions[1], motions[2])
    pred_motion = motion_ops.append(motions[1], motions[0])
    bvh.save(
        ref_motion, save_output_path.joinpath("ref", f"{i}.bvh"),
    )
    bvh.save(
        pred_motion, save_output_path.joinpath("pred", f"{i}.bvh"),
    )


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


def save_motion_files(seqs_T: List[np.ndarray], save_output_path: Path):
    idxs_to_save = [i for i in range(0, len(seqs_T[0]), len(seqs_T[0]) // 10)]
    amass_dip_motion = amass_dip.load(
        file=None, load_skel=True, load_motion=False,
    )

    fairmotion_utils.create_dir_if_absent(save_output_path.joinpath("ref"))
    fairmotion_utils.create_dir_if_absent(save_output_path.joinpath("pred"))

    pool = Pool(10)
    indices = range(len(seqs_T[0]))
    skels = [amass_dip_motion.skel for _ in indices]
    save_output_paths = [save_output_path for _ in indices]
    pool.starmap(
        save_seq, [list(zip(indices, *seqs_T, skels, save_output_paths))[i] for i in idxs_to_save]
    )


def calculate_metrics(
    pred_seqs: np.ndarray,
    tgt_seqs: np.ndarray
) -> Dict[int, np.float32]:

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
    max_len: int,
    src_len: int
) -> Tuple[List[np.ndarray], Dict[int, np.float32]]:
    pred_seqs, src_seqs, tgt_seqs = run_model(
        model, dataset, max_len, device, mean, std, src_len
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
    LOGGER.info(f"Using device: {device}")

    LOGGER.info("Preparing dataset")
    dataset, mean, std = utils.prepare_dataset(
        *[
            args.preprocessed_path.joinpath(f"{split}.pkl")
            for split in ["train", "test", "validation"]
        ],
        batch_size=args.batch_size,
        device=device,
        shuffle=args.shuffle,
    )
    # number of predictions per time step = num_joints * angle representation
    data_shape = next(iter(dataset["train"]))[0].shape
    num_predictions = data_shape[-1]

    LOGGER.info("Preparing model")
    model_filename = f"{args.epoch if args.epoch else 'best'}.model"
    model_file_path = args.save_model_path.joinpath(model_filename)
    model = prepare_model(
        model_file_path,
        num_predictions=num_predictions,
        hidden_dim=args.hidden_dim,
        device=device,
        num_layers=args.num_layers,
        architecture=args.architecture,
        src_len=args.src_len,
        num_heads = args.num_heads,
        ninp = args.ninp,
        num_experts=args.num_experts
    )

    LOGGER.info("Running model")
    rep = args.preprocessed_path.name

    seqs_T, mae = test_model(
        model, dataset["test"], rep, device, mean, std, args.max_len, args.src_len
    )
    LOGGER.info(
        "Test MAE: "
        + " | ".join([f"{frame}: {mae[frame]}" for frame in mae.keys()])
    )

    if args.save_output_path:
        LOGGER.info("Saving results")
        save_motion_files(seqs_T, args.save_output_path)


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
        "--save-output-path",
        dest="save_output_path",
        # type=str,
        type=lambda p: Path(p).expanduser().resolve(strict=True),
        help="Path to store predicted motion",
        default=None,
    )
    parser.add_argument(
        "--src_len",
        dest="src_len",
        type=int,
        help="Input size for predictions. Needs to match what model is expecting from training.",
        default=120
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
        "--epoch",
        type=int,
        help="Model from epoch to test, will test on best"
        " model if not specified",
        default=None,
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
