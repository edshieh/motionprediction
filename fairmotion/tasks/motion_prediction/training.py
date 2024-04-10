# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import sys
from pathlib import Path
from shutil import rmtree

from fairmotion.models import (  # Used for typing
    rnn,
    seq2seq,
    transformer,
    SpatioTemporalTransformer,
    moe
)
from fairmotion.tasks.motion_prediction import generate, utils, test
from fairmotion.utils import utils as fairmotion_utils

from typing import Any, Callable, Dict, Iterable, List, Tuple


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

    log_file = model_save_path.joinpath(f"training_{model_architecture}.log")
    if log_file.exists():
        log_file.unlink()
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(logFormatter)
    LOGGER.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    LOGGER.addHandler(consoleHandler)

def set_seeds(device):
    if device == "mps":
        torch.mps.manual_seed(1)
    else:
        torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_initial_epoch_and_validation_loss(
        model: (
            rnn.RNN |
            seq2seq.Seq2Seq |
            seq2seq.TiedSeq2Seq |
            transformer.TransformerLSTMModel |
            transformer.TransformerModel |
            SpatioTemporalTransformer.TransformerSpatialTemporalModel |
            moe.moe
        ),
        dataset: Dict,
        device: str,
        criterion: nn.MSELoss,
        num_training_sequences: int,
        batch_size: int
    ):
    epoch_loss = 0

    for _, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
        with torch.no_grad():
            model.eval()
        src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
        outputs = model(src_seqs, tgt_seqs, teacher_forcing_ratio=1,)
        loss = criterion(outputs, tgt_seqs)
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / num_training_sequences
    val_loss = generate.eval(
        model, criterion, dataset["validation"], batch_size, device,
    )
    return epoch_loss, val_loss


def train(args: argparse.Namespace):
    args.device = fairmotion_utils.set_device(args.device) # reset this so the logged config shows the device used
    LOGGER.info(args._get_kwargs())
    utils.log_config(args.save_model_path, args._get_kwargs())

    device = args.device
    LOGGER.info(f"Using device: {device}")

    set_seeds(device)

    LOGGER.info("Preparing dataset...")
    dataset, mean, std = utils.prepare_dataset(
        *[
            args.preprocessed_path.joinpath(f"{split}.pkl")
            for split in ["train", "test", "validation"]
        ],
        batch_size=args.batch_size,
        device=device,
        shuffle=args.shuffle,
    )

    if device == "mps":
        LOGGER.info(
            "MPS Current allocated memory after data load: "
            f"{fairmotion_utils.convert_byte_to_humanreadable(torch.mps.current_allocated_memory())} bytes"
        )
        LOGGER.info(
            "MPS Driver allocated memory after data load: "
            f"{fairmotion_utils.convert_byte_to_humanreadable(torch.mps.driver_allocated_memory())} bytes"
        )

    # Loss per epoch is the average loss per sequence
    num_training_sequences = len(dataset["train"]) * args.batch_size

    # number of predictions per time step = num_joints * angle representation
    # shape is (batch_size, seq_len, num_predictions)
    _, tgt_len, num_predictions = next(iter(dataset["train"]))[1].shape

    # Load in the desired model
    model = utils.prepare_model(
        input_dim=num_predictions,
        hidden_dim=args.hidden_dim,
        device=device,
        num_layers=args.num_layers,
        architecture=args.architecture,
        num_heads = args.num_heads,
        src_len=120,
        ninp = args.ninp
    )

    if device == "mps":
        LOGGER.info(
            "MPS Current allocated memory after model load: "
            f"{fairmotion_utils.convert_byte_to_humanreadable(torch.mps.current_allocated_memory())} bytes"
        )
        LOGGER.info(
            "MPS Driver allocated memory after model load: "
            f"{fairmotion_utils.convert_byte_to_humanreadable(torch.mps.driver_allocated_memory())} bytes"
        )

    criterion = nn.MSELoss()
    model.init_weights()
    training_losses, val_losses = [], []

    # Log model loss before any training
    # epoch_loss, val_loss = get_initial_epoch_and_validation_loss(
    #     model, dataset, device, criterion, num_training_sequences, args.batch_size
    # )
    # LOGGER.info(
    #     "Before training: "
    #     f"Training loss {epoch_loss} | "
    #     f"Validation loss {val_loss}"
    # )

    LOGGER.info("Training model...")
    torch.autograd.set_detect_anomaly(True)
    opt = utils.prepare_optimizer(model, args.optimizer, args.lr)
    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()

        # Use a dummy value for the calculation if tuning is True, else use the actual epoch count
        effective_epochs = 100 if args.tuning else args.epochs

        teacher_forcing_ratio = np.clip(
            (1 - 2 * epoch / effective_epochs), a_min=0, a_max=1,
        )
        LOGGER.info(
            f"Running epoch {epoch} | "
            f"teacher_forcing_ratio={teacher_forcing_ratio}"
        )

        num_iterations = len(dataset["train"])
        for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
            # print(f"Iteration: {iterations}/{num_iterations}")

            opt.optimizer.zero_grad()
            src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
            # if device == "mps":
            #     LOGGER.info(
            #         "MPS Current allocated memory after tensor to device: "
            #         f"{fairmotion_utils.convert_byte_to_humanreadable(torch.mps.driver_allocated_memory())} bytes"
            #     )

            # LOGGER.info("Forward pass...")
            # print(src_seqs.is_contiguous(memory_format=torch.channels_last))
            # print(tgt_seqs.is_contiguous(memory_format=torch.channels_last))

            if args.architecture == "moe":
                outputs, total_aux_loss = model(
                    src_seqs, tgt_seqs, teacher_forcing_ratio=teacher_forcing_ratio
                )
            else:
                outputs = model(
                    src_seqs, tgt_seqs, teacher_forcing_ratio=teacher_forcing_ratio
                )
                total_aux_loss = 0

            # print(outputs.is_contiguous(memory_format=torch.channels_last))
            if device == "mps":
                outputs = outputs.float()
            else:
                outputs = outputs.double()

            # if device == "mps":
            #     LOGGER.info(
            #         "MPS Current allocated memory after forward pass: "
            #         f"{fairmotion_utils.convert_byte_to_humanreadable(torch.mps.driver_allocated_memory())} bytes"
            #     )
            # LOGGER.info("Calculate loss...")

            # x = utils.prepare_tgt_seqs(args.architecture, src_seqs, tgt_seqs)
            # print(torch.isnan(x).any(), torch.isinf(x).any())
            # print(torch.isnan(outputs).any(), torch.isinf(outputs).any())

            # Calculate the main loss
            main_loss = criterion(
                outputs,
                utils.prepare_tgt_seqs(args.architecture, src_seqs, tgt_seqs),
            )
            loss = main_loss + total_aux_loss

            # if device == "mps":
            #     LOGGER.info(
            #         "MPS Current allocated memory after loss: "
            #         f"{fairmotion_utils.convert_byte_to_humanreadable(torch.mps.driver_allocated_memory())} bytes"
            #     )
            # LOGGER.info("Backward pass...")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            # print(loss.is_contiguous(memory_format=torch.channels_last))
            # LOGGER.info(f"MPS Current allocated memory after backward pass: {torch.mps.driver_allocated_memory()} bytes")
            # LOGGER.info("Opt step...")
            opt.step()
            epoch_loss += loss.item()

            # if device == "mps":
            #     LOGGER.info(
            #         "MPS Current allocated memory after opt step: "
            #         f"{fairmotion_utils.convert_byte_to_humanreadable(torch.mps.driver_allocated_memory())} bytes"
            #     )

        epoch_loss = epoch_loss / num_training_sequences
        training_losses.append(epoch_loss)
        val_loss = generate.eval(
            model, criterion, dataset["validation"], args.batch_size, device,
        )
        val_losses.append(val_loss)
        opt.epoch_step(val_loss=val_loss)
        LOGGER.info(
            f"Training loss {epoch_loss} | "
            f"Validation loss {val_loss} | "
            f"Iterations {iterations + 1}"
        )

        # Get validation MAE if we are at save frequency and the final
        if epoch % args.save_model_frequency == 0 or epoch + 1 == args.epochs:
            rep = args.preprocessed_path.name
            _, mae = test.test_model(
                model=model,
                dataset=dataset["validation"],
                rep=rep,
                device=device,
                mean=mean,
                std=std,
                max_len=tgt_len,
            )
            LOGGER.info(f"Validation MAE: {mae}")
            torch.save(
                model.state_dict(), str(args.save_model_path.joinpath(f"{epoch}.model"))
            )
            if len(val_losses) == 0 or val_loss <= min(val_losses):
                torch.save(
                    model.state_dict(), str(args.save_model_path.joinpath("best.model"))
                )
    return training_losses, val_losses


def plot_curves(args: argparse.Namespace, training_losses: List[float], val_losses: List[float]):
    plt.plot(range(len(training_losses)), training_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(args.save_model_path.joinpath("loss.svg"), format="svg")


def validate_args(args: argparse.Namespace):
    # validate provided options
    if not args.preprocessed_path.is_dir():
        raise ValueError(f"Value given for '--prepocessed-path' must be a directory. Given: {args.preprocessed_path}")

    if args.save_model_path.exists():
        if args.force:
            print(f"'Force' enabled. Removing directory {args.save_model_path} without user input.")
            rmtree(args.save_model_path)
        else:
            yes_no_response = fairmotion_utils.yes_no_input(f"Directory {args.save_model_path} already exists. \nDo you want to delete? (yes/y or no/n): ")
            if yes_no_response:
                print(f"Removing directory {args.save_model_path} with user input.")
                rmtree(args.save_model_path)
            else:
                print("Not deleting directory. Please change config value for '--save-model-path'\n")
                sys.exit()

def main(args: argparse.Namespace):
    validate_args(args)
    fairmotion_utils.create_dir_if_absent(args.save_model_path)
    configure_logging(args.architecture, args.save_model_path)
    train_losses, val_losses = train(args)
    plot_curves(args, train_losses, val_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequence to sequence motion prediction training"
    )
    parser.add_argument(
        "--preprocessed-path",
        dest="preprocessed_path",
        type=lambda p: Path(p).expanduser().resolve(strict=True),
        help="Path to folder with pickled files",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="Batch size for training",
        default=64
    )
    parser.add_argument(
        "--shuffle", action='store_true',
        help="Use this option to enable shuffling",
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
        "--dropout",
        type=float,
        help="Dropout rate",
        default=0.1,
    )
    parser.add_argument(
        "--save-model-path",
        dest="save_model_path",
        type=lambda p: Path(p).expanduser().resolve(strict=False),
        help="Path to store saved models",
        required=True,
    )
    parser.add_argument(
        "--save-model-frequency",
        dest="save_model_frequency",
        type=int,
        help="Frequency (in terms of number of epochs) at which model is "
        "saved",
        default=5,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
        default=200
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
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
        default=None,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Torch optimizer",
        default="sgd",
        choices=["adam", "sgd", "noamopt"],
    )
    parser.add_argument(
        "-f", "--force",
        dest="force",
        action="store_true",
        default=False,
        help="Enable force for any stdin required during consolidation process"
    )

    parser.add_argument(
        "-t", "--tuning",
        dest="tuning",
        action="store_true",
        default=False,
        help="Enable tuning option which keeps tfr as if epochs were 100"
    )

    args = parser.parse_args()

    main(args)
