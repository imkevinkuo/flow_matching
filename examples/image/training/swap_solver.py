# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""
Solvers that support label swapping during sampling.
"""

from typing import Callable, List, Optional, Sequence, Union

import torch
from torch import Tensor

from flow_matching.solver.ode_solver import ODESolver
from flow_matching.solver.discrete_solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper


def parse_swap_steps(swap_steps_file: str) -> List[float]:
    """
    Parse swap_steps from a text file containing comma-separated decimal fractions.

    Args:
        swap_steps_file: Path to text file containing comma-separated fractions like "0.4,0.6,0.3"

    Returns:
        List of floats representing fraction of total steps for each swap period

    Raises:
        ValueError: If file cannot be read, parsed, or contains invalid values
        FileNotFoundError: If file does not exist
    """
    if swap_steps_file is None or swap_steps_file.strip() == "":
        raise ValueError("swap_steps file path cannot be None or empty")

    try:
        with open(swap_steps_file, 'r') as f:
            content = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"swap_steps file not found: {swap_steps_file}")
    except Exception as e:
        raise ValueError(f"Error reading swap_steps file {swap_steps_file}: {e}") from e

    if not content:
        raise ValueError(f"swap_steps file is empty: {swap_steps_file}")

    try:
        fractions = [float(s.strip()) for s in content.split(',')]
    except ValueError as e:
        raise ValueError(f"swap_steps file must contain comma-separated decimal numbers, got: {content}") from e

    if any(f <= 0 for f in fractions):
        raise ValueError(f"All swap_steps fractions must be positive, got: {fractions}")

    total = sum(fractions)
    if total > 1.0:
        raise ValueError(f"Sum of swap_steps fractions must be <= 1.0, got sum={total} for fractions: {fractions}")

    return fractions


def convert_fractions_to_steps(fractions: List[float], total_steps: int) -> List[int]:
    """
    Convert fraction list to integer step counts.

    Args:
        fractions: List of decimal fractions (e.g., [0.4, 0.6, 0.3])
        total_steps: Total number of denoising steps

    Returns:
        List of integer step counts

    Raises:
        ValueError: If total_steps is not positive
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got: {total_steps}")

    return [max(1, round(f * total_steps)) for f in fractions]


class LabelSwappingODESolver(ODESolver):
    """
    ODE Solver that swaps labels according to a schedule during sampling.

    This solver wraps the standard ODESolver and modifies the label conditioning
    at specific timesteps according to the swap_steps schedule.
    """

    def __init__(self, velocity_model: Union[ModelWrapper, Callable]):
        super().__init__(velocity_model)

    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
        swap_steps: Optional[List[float]] = None,
        label: Optional[Tensor] = None,
        label_other: Optional[Tensor] = None,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        """
        Sample with label swapping support.

        Args:
            x_init: Initial conditions
            step_size: The step size (required for fixed-step methods like euler/midpoint)
            method: ODE solver method
            atol: Absolute tolerance
            rtol: Relative tolerance
            time_grid: Time discretization
            return_intermediates: Whether to return intermediate steps
            enable_grad: Whether to enable gradients
            swap_steps: List of step fractions for swapping (e.g., [0.4, 0.6, 0.3])
            label: Initial label tensor
            label_other: Alternative label to swap to
            **model_extras: Additional model arguments

        Returns:
            Final sample or sequence of samples if return_intermediates=True
        """
        if swap_steps is None or label_other is None:
            # No swapping, use standard sampling
            return super().sample(
                x_init=x_init,
                step_size=step_size,
                method=method,
                atol=atol,
                rtol=rtol,
                time_grid=time_grid,
                return_intermediates=return_intermediates,
                enable_grad=enable_grad,
                label=label,
                **model_extras,
            )

        # Determine total number of steps based on method and step_size
        if step_size is None:
            raise ValueError("swap_steps requires fixed step_size (euler/midpoint/heun3)")

        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()
        total_steps = int((t_final - t_init) / step_size)

        # Convert fractions to integer step counts
        swap_steps_int = convert_fractions_to_steps(swap_steps, total_steps)

        # Build schedule: which label to use at each step
        current_label = label
        step_counter = 0
        swap_index = 0
        use_initial = True  # Start with initial label

        results = []
        x_current = x_init

        for i in range(total_steps):
            # Check if we need to swap
            if swap_index < len(swap_steps_int) and step_counter >= swap_steps_int[swap_index]:
                # Swap labels
                use_initial = not use_initial
                current_label = label if use_initial else label_other
                swap_index += 1
                step_counter = 0

            # Take a single step with current label
            t_start = t_init + i * step_size
            t_end = min(t_start + step_size, t_final)
            local_time_grid = torch.tensor([t_start, t_end], device=x_init.device)

            x_current = super().sample(
                x_init=x_current,
                step_size=step_size,
                method=method,
                atol=atol,
                rtol=rtol,
                time_grid=local_time_grid,
                return_intermediates=False,
                enable_grad=enable_grad,
                label=current_label,
                **model_extras,
            )

            step_counter += 1

            if return_intermediates:
                results.append(x_current)

        if return_intermediates:
            return results
        else:
            return x_current


class LabelSwappingDiscreteEulerSolver(MixtureDiscreteEulerSolver):
    """
    Discrete Euler Solver that swaps labels according to a schedule during sampling.

    This solver wraps the standard MixtureDiscreteEulerSolver and modifies the label
    conditioning at specific timesteps according to the swap_steps schedule.
    """

    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        swap_steps: Optional[List[float]] = None,
        label: Optional[Tensor] = None,
        label_other: Optional[Tensor] = None,
        **model_extras,
    ) -> Tensor:
        """
        Sample with label swapping support.

        Args:
            x_init: Initial state
            step_size: Step size for uniform discretization
            div_free: Divergence-free coefficient
            dtype_categorical: Precision for categorical sampler
            time_grid: Time interval
            return_intermediates: Whether to return intermediate steps
            verbose: Whether to print progress
            swap_steps: List of step fractions for swapping (e.g., [0.4, 0.6, 0.3])
            label: Initial label tensor
            label_other: Alternative label to swap to
            **model_extras: Additional model arguments

        Returns:
            Final sampled sequence
        """
        if swap_steps is None or label_other is None:
            # No swapping, use standard sampling
            return super().sample(
                x_init=x_init,
                step_size=step_size,
                div_free=div_free,
                dtype_categorical=dtype_categorical,
                time_grid=time_grid,
                return_intermediates=return_intermediates,
                verbose=verbose,
                label=label,
                **model_extras,
            )

        if step_size is None:
            raise ValueError("swap_steps requires explicit step_size")

        from math import ceil
        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()
        total_steps = ceil((t_final - t_init) / step_size)

        # Convert fractions to integer step counts
        swap_steps_int = convert_fractions_to_steps(swap_steps, total_steps)

        # Build schedule
        current_label = label
        step_counter = 0
        swap_index = 0
        use_initial = True

        x_current = x_init

        for i in range(total_steps):
            # Check if we need to swap
            if swap_index < len(swap_steps_int) and step_counter >= swap_steps_int[swap_index]:
                use_initial = not use_initial
                current_label = label if use_initial else label_other
                swap_index += 1
                step_counter = 0

            # Take a single step with current label
            t_start = t_init + i * step_size
            t_end = min(t_start + step_size, t_final)
            local_time_grid = torch.tensor([t_start, t_end], device=x_init.device)

            x_current = super().sample(
                x_init=x_current,
                step_size=step_size,
                div_free=div_free,
                dtype_categorical=dtype_categorical,
                time_grid=local_time_grid,
                return_intermediates=False,
                verbose=False,
                label=current_label,
                **model_extras,
            )

            step_counter += 1

        return x_current
