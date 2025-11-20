# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import logging

from models.model_configs import MODEL_CONFIGS
from torchdiffeq._impl.odeint import SOLVERS

logger = logging.getLogger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser("Image dataset training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=921, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--optimizer_betas",
        nargs="+",
        type=float,
        default=[0.9, 0.95],
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--decay_lr",
        action="store_true",
        help="Adds a linear decay to the lr during training.",
    )
    parser.add_argument(
        "--class_drop_prob",
        type=float,
        default=0.2,
        help="Probability to drop conditioning during training",
    )
    parser.add_argument(
        "--skewed_timesteps",
        action="store_true",
        help="Use skewed timestep sampling proposed in the EDM paper: https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--edm_schedule",
        action="store_true",
        help="Use the alternative time discretization during sampling proposed in the EDM paper: https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="When evaluating, use the model Exponential Moving Average weights.",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        # default=list(MODEL_CONFIGS.keys())[0],
        type=str,
        # choices=list(MODEL_CONFIGS.keys()),
        help="comma-separated list of datasets to use.",
    )
    parser.add_argument(
        "--data_path",
        default="/data/matrix/projects/smith/kkuo2/concept_composition/datasets",
        type=str,
        help="Base dataset path",
    )
    parser.add_argument(
        "--image_size",
        default=None,
        type=int,
        help="Resize images to this size (e.g., 64 for 64x64). If None, images are loaded at their native resolution.",
    )
    parser.add_argument(
        "--captions",
        default="false",
        type=str,
        choices=["true", "false"],
        help="Enable caption loading from JSON files for text conditioning",
    )
    parser.add_argument(
        "--num_classes",
        default=None,
        type=int,
        help="Number of classes for class conditioning (1 = unconditional, >1 = conditional). If None, auto-detected from dataset.",
    )
    parser.add_argument(
        "--train_folder",
        default="train",
        type=str,
        help="Name of the training folder within each dataset",
    )
    parser.add_argument(
        "--val_folder",
        default="test",
        type=str,
        help="Name of the validation folder within each dataset",
    )
    parser.add_argument(
        "--max_examples_per_dataset",
        default=None,
        type=int,
        help="Maximum number of examples to use from each dataset. If None, use all examples.",
    )
    parser.add_argument(
        "--celeba_attributes",
        default="",
        type=str,
        help="Comma-separated list of CelebA attributes for conditioning.",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--ode_method",
        default="midpoint",
        choices=list(SOLVERS.keys()) + ["edm_heun"],
        help="ODE solver used to generate samples.",
    )
    parser.add_argument(
        "--ode_options",
        default='{"step_size": 0.01}',
        type=json.loads,
        help="ODE solver options. Eg. the midpoint solver requires step-size, dopri5 has no options to set.",
    )
    parser.add_argument(
        "--sym",
        default=0.0,
        type=float,
        help="Symmetric term for sampling the discrete flow.",
    )
    parser.add_argument(
        "--temp",
        default=1.0,
        type=float,
        help="Temperature for sampling the discrete flow.",
    )
    parser.add_argument(
        "--sym_func",
        action="store_true",
        help="Use a fixed function for the symmetric term in the discrete flow.",
    )
    parser.add_argument(
        "--sampling_dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Solver dtype for sampling the discrete flow.",
    )
    parser.add_argument(
        "--cfg_scale",
        default=0.2,
        type=float,
        help="Classifier-free guidance scale for generating samples.",
    )
    parser.add_argument(
        "--fid_samples",
        default=50000,
        type=int,
        help="number of synthetic samples for FID evaluations",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="start epoch (used when resumed from checkpoint)",
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="No training, only run evaluation"
    )
    parser.add_argument(
        "--eval_frequency",
        default=50,
        type=int,
        help="Frequency (in number of epochs) for running FID evaluation. -1 to never run evaluation.",
    )
    parser.add_argument(
        "--compute_fid",
        action="store_true",
        help="Whether to compute FID in the evaluation loop. When disabled, the evaluation loop still runs and saves snapshots, but skips the FID computation.",
    )
    parser.add_argument(
        "--save_fid_samples",
        action="store_true",
        help="Save all samples generated for FID computation.",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Only run one batch of training and evaluation.",
    )
    parser.add_argument(
        "--discrete_flow_matching",
        action="store_true",
        help="Train discrete flow matching model.",
    )
    parser.add_argument(
        "--discrete_fm_steps",
        default=1024,
        type=int,
        help="Number of sampling steps for discrete FM.",
    )
    parser.add_argument(
        "--swap_steps",
        default=None,
        type=str,
        help="Path to text file containing comma-separated decimal fractions for swapping conditioning labels during evaluation. "
             "Fractions must sum to 1.0 or less. Example file content: '0.3,0.4,0.2' (sum=0.9) will use initial label for 30%% of total steps, "
             "swap to other label for 40%% of steps, swap back to initial label for 20%% of steps, then use other label for remaining 10%% of steps. "
             "The fractions are converted to integer steps by multiplying by total steps and rounding.",
    )

    return parser
