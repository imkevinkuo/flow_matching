# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import PIL.Image

import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.utils import ModelWrapper
from models.discrete_unet import DiscreteUNetModel
from models.ema import EMA
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from training import distributed_mode
from training.edm_time_discretization import get_time_discretization
from training.train_loop import MASK_TOKEN
from training.swap_solver import (
    parse_swap_steps,
    convert_fractions_to_steps,
    LabelSwappingODESolver,
    LabelSwappingDiscreteEulerSolver,
)

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float, label: torch.Tensor
    ):
        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )
        is_discrete = isinstance(module, DiscreteUNetModel) or (
            isinstance(module, EMA) and isinstance(module.model, DiscreteUNetModel)
        )
        assert (
            cfg_scale == 0.0 or not is_discrete
        ), f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            with torch.cuda.amp.autocast(), torch.no_grad():
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})
            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
        else:
            # Model is fully conditional, no cfg weighting needed
            with torch.cuda.amp.autocast(), torch.no_grad():
                result = self.model(x, t, extra={"label": label})

        self.nfe_counter += 1
        if is_discrete:
            return torch.softmax(result.to(dtype=torch.float32), dim=-1)
        else:
            return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
    clip_encoder=None,
):
    gc.collect()
    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)

    # Parse swap_steps if provided
    swap_steps_fractions = None
    if hasattr(args, 'swap_steps') and args.swap_steps is not None:
        swap_steps_fractions = parse_swap_steps(args.swap_steps)
        logger.info(f"Using swap_steps fractions from file {args.swap_steps}: {swap_steps_fractions}")

        # Calculate and log the actual step counts for both discrete and continuous modes
        if args.discrete_flow_matching:
            total_steps = args.discrete_fm_steps
        else:
            ode_opts = args.ode_options
            step_size = ode_opts.get("step_size", 0.01)
            total_steps = int(1.0 / step_size)

        swap_steps_int = convert_fractions_to_steps(swap_steps_fractions, total_steps)
        logger.info(f"Converted to integer steps (total={total_steps}): {swap_steps_int}")

    if args.discrete_flow_matching:
        scheduler = PolynomialConvexScheduler(n=3.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        p = torch.zeros(size=[257], dtype=torch.float32, device=device)
        p[256] = 1.0
        if swap_steps_fractions is not None:
            solver = LabelSwappingDiscreteEulerSolver(
                model=cfg_scaled_model,
                path=path,
                vocabulary_size=257,
                source_distribution_p=p,
            )
        else:
            solver = MixtureDiscreteEulerSolver(
                model=cfg_scaled_model,
                path=path,
                vocabulary_size=257,
                source_distribution_p=p,
            )
    else:
        if swap_steps_fractions is not None:
            solver = LabelSwappingODESolver(velocity_model=cfg_scaled_model)
        else:
            solver = ODESolver(velocity_model=cfg_scaled_model)
        ode_opts = args.ode_options

    fid_metric = FrechetInceptionDistance(normalize=True).to(
        device=device, non_blocking=True
    )

    num_synthetic = 0
    # Track which snapshot groups have been saved
    snapshots_saved = {
        'unconditional': False,
        'label_0': False,
        'label_1': False,
        'average': False,
    }
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    # Create an iterator from the data loader for cycling through batches
    data_iter = iter(data_loader)
    data_iter_step = 0

    # Continue generating until we have enough synthetic samples
    while num_synthetic < fid_samples:
        logger.info(f"[DEBUG] Processing batch {data_iter_step}, num_synthetic={num_synthetic}/{fid_samples}")

        # Get next batch, cycling back to beginning if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            logger.info(f"[DEBUG] Data loader exhausted, restarting from beginning")
            data_iter = iter(data_loader)
            batch = next(data_iter)

        # Handle both 2-tuple (image, label) and 3-tuple (image, label, caption)
        if len(batch) == 3:
            samples, labels, captions = batch
        else:
            samples, labels = batch
            captions = None

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logger.info(f"[DEBUG] Batch shape: {samples.shape}, labels shape: {labels.shape}")
        fid_metric.update(samples, real=True)

        if num_synthetic < fid_samples:
            # Determine which evaluation group this batch belongs to based on swap_steps
            if swap_steps_fractions is not None:
                # Split batches into 4 groups
                samples_per_group = fid_samples // 4
                current_group_idx = num_synthetic // samples_per_group
                logger.info(f"[DEBUG] samples_per_group={samples_per_group}, current_group_idx={current_group_idx}")

                if current_group_idx == 0:
                    # Group 1: Unconditional generation
                    group_name = 'unconditional'
                    generation_labels = None
                    label_other = None
                    use_swap = False
                elif current_group_idx == 1:
                    # Group 2: Start with label 0, swap with label 1
                    group_name = 'label_0'
                    batch_size = samples.shape[0]
                    generation_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
                    label_other = torch.ones(batch_size, dtype=torch.long, device=device)
                    use_swap = True
                elif current_group_idx == 2:
                    # Group 3: Start with label 1, swap with label 0
                    group_name = 'label_1'
                    batch_size = samples.shape[0]
                    generation_labels = torch.ones(batch_size, dtype=torch.long, device=device)
                    label_other = torch.zeros(batch_size, dtype=torch.long, device=device)
                    use_swap = True
                else:
                    # Group 4: Average of label embeddings
                    group_name = 'average'
                    batch_size = samples.shape[0]
                    # Use special marker for label averaging if supported
                    if hasattr(args, 'num_classes') and args.num_classes is not None and args.num_classes > 1:
                        generation_labels = torch.full((batch_size,), args.num_classes, dtype=torch.long, device=device)
                    else:
                        # Fallback: just use label 0
                        generation_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
                    label_other = None
                    use_swap = False
            else:
                # No swap_steps: use original behavior
                generation_labels = labels.clone()
                if hasattr(args, 'num_classes') and args.num_classes is not None and args.num_classes > 1:
                    generation_labels[-1] = args.num_classes
                label_other = None
                use_swap = False
                group_name = 'default'

            logger.info(f"[DEBUG] Generating group '{group_name}', use_swap={use_swap}, generation_labels shape={generation_labels.shape if generation_labels is not None else None}")

        if num_synthetic < fid_samples:
            cfg_scaled_model.reset_nfe_counter()
            if args.discrete_flow_matching:
                # Discrete sampling
                x_0 = (
                    torch.zeros(samples.shape, dtype=torch.long, device=device)
                    + MASK_TOKEN
                )
                if args.sym_func:
                    sym = lambda t: 12.0 * torch.pow(t, 2.0) * torch.pow(1.0 - t, 0.25)
                else:
                    sym = args.sym
                if args.sampling_dtype == "float32":
                    dtype = torch.float32
                elif args.sampling_dtype == "float64":
                    dtype = torch.float64

                if use_swap:
                    synthetic_samples = solver.sample(
                        x_init=x_0,
                        step_size=1.0 / args.discrete_fm_steps,
                        verbose=False,
                        div_free=sym,
                        dtype_categorical=dtype,
                        label=generation_labels,
                        label_other=label_other,
                        swap_steps=swap_steps_fractions,
                        cfg_scale=args.cfg_scale,
                    )
                else:
                    synthetic_samples = solver.sample(
                        x_init=x_0,
                        step_size=1.0 / args.discrete_fm_steps,
                        verbose=False,
                        div_free=sym,
                        dtype_categorical=dtype,
                        label=generation_labels,
                        cfg_scale=args.cfg_scale,
                    )
            else:
                # Continuous sampling
                x_0 = torch.randn(samples.shape, dtype=torch.float32, device=device)

                if args.edm_schedule:
                    time_grid = get_time_discretization(nfes=ode_opts["nfe"])
                else:
                    time_grid = torch.tensor([0.0, 1.0], device=device)

                if use_swap:
                    synthetic_samples = solver.sample(
                        time_grid=time_grid,
                        x_init=x_0,
                        method=args.ode_method,
                        return_intermediates=False,
                        atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                        rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
                        step_size=ode_opts["step_size"]
                        if "step_size" in ode_opts
                        else None,
                        label=generation_labels,
                        label_other=label_other,
                        swap_steps=swap_steps_fractions,
                        cfg_scale=args.cfg_scale,
                    )
                else:
                    synthetic_samples = solver.sample(
                        time_grid=time_grid,
                        x_init=x_0,
                        method=args.ode_method,
                        return_intermediates=False,
                        atol=ode_opts["atol"] if "atol" in ode_opts else 1e-5,
                        rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-5,
                        step_size=ode_opts["step_size"]
                        if "step_size" in ode_opts
                        else None,
                        label=generation_labels,
                        cfg_scale=args.cfg_scale,
                    )

                # Scaling to [0, 1] from [-1, 1]
                synthetic_samples = torch.clamp(
                    synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
                )
                synthetic_samples = torch.floor(synthetic_samples * 255)
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
            logger.info(
                f"{samples.shape[0]} samples generated in {cfg_scaled_model.get_nfe()} evaluations."
            )
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                logger.info(f"[DEBUG] Truncating samples: {num_synthetic + synthetic_samples.shape[0]} > {fid_samples}, keeping {fid_samples - num_synthetic}")
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            fid_metric.update(synthetic_samples, real=False)
            num_synthetic += synthetic_samples.shape[0]
            logger.info(f"[DEBUG] Updated num_synthetic to {num_synthetic}/{fid_samples}")

            # Save snapshots for each group
            if swap_steps_fractions is not None and args.output_dir:
                if not snapshots_saved[group_name]:
                    save_image(
                        synthetic_samples,
                        fp=Path(args.output_dir)
                        / "snapshots"
                        / f"{epoch}_{group_name}.png",
                    )
                    snapshots_saved[group_name] = True
                    logger.info(f"Saved snapshot for group: {group_name}")
            elif not any(snapshots_saved.values()) and args.output_dir:
                # Original behavior when swap_steps is not used
                save_image(
                    synthetic_samples,
                    fp=Path(args.output_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}.png",
                )
                snapshots_saved['default'] = True

            if args.save_fid_samples and args.output_dir:
                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for batch_index, image_np in enumerate(images_np):
                    image_dir = Path(args.output_dir) / "fid_samples"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = (
                        image_dir
                        / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
                    )
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        if data_iter_step % PRINT_FREQUENCY == 0:
            # Sync fid metric to ensure that the processes dont deviate much.
            gc.collect()
            running_fid = fid_metric.compute()
            logger.info(
                f"Evaluating [batch {data_iter_step}] samples generated [{num_synthetic}/{fid_samples}] running fid {running_fid}"
            )

        data_iter_step += 1

        if args.test_run:
            logger.info(f"[DEBUG] Breaking due to test_run")
            break

    logger.info(f"[DEBUG] Exited generation loop. Total batches processed: {data_iter_step}, num_synthetic={num_synthetic}/{fid_samples}")

    if not args.compute_fid:
        return {}

    return {"fid": float(fid_metric.compute().detach().cpu())}
