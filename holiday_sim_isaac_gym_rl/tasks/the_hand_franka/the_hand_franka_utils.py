
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, List

from torch import Tensor


@dataclass
class DofParameters:
    """Joint/dof parameters."""
    the_hand_stiffness: float
    franka_stiffness: float
    the_hand_effort: float
    franka_effort: List[float]  # separate per DOF
    the_hand_damping: float
    franka_damping: float
    dof_friction: float
    the_hand_armature: float
    franka_armature: float

    @staticmethod
    def from_cfg(cfg: Dict) -> DofParameters:
        return DofParameters(
            the_hand_stiffness=cfg["env"]["theHandStiffness"],
            franka_stiffness=cfg["env"]["frankaStiffness"],
            the_hand_effort=cfg["env"]["theHandEffort"],
            franka_effort=cfg["env"]["frankaEffort"],
            the_hand_damping=cfg["env"]["theHandDamping"],
            franka_damping=cfg["env"]["frankaDamping"],
            dof_friction=cfg["env"]["dofFriction"],
            the_hand_armature=cfg["env"]["theHandArmature"],
            franka_armature=cfg["env"]["frankaArmature"],
        )


def populate_dof_properties(hand_arm_dof_props, params: DofParameters, arm_dofs: int, hand_dofs: int) -> None:
    assert len(hand_arm_dof_props["stiffness"]) == arm_dofs + hand_dofs

    # hand_arm_dof_props["stiffness"][0:arm_dofs].fill(params.franka_stiffness)
    # hand_arm_dof_props["stiffness"][arm_dofs:].fill(params.gripper_stiffness)
    
    assert len(params.franka_effort) == arm_dofs

    # hand_arm_dof_props["effort"][0:arm_dofs] = params.franka_effort
    # hand_arm_dof_props["effort"][arm_dofs:].fill(params.gripper_effort)

    # hand_arm_dof_props["damping"][0:arm_dofs].fill(params.franka_damping)
    # hand_arm_dof_props["damping"][arm_dofs:].fill(params.gripper_damping)

    # if params.dof_friction >= 0:
    #     hand_arm_dof_props["friction"].fill(params.dof_friction)

    # hand_arm_dof_props["armature"][0:arm_dofs].fill(params.franka_armature)
    # hand_arm_dof_props["armature"][arm_dofs:].fill(params.gripper_armature)

def tolerance_curriculum(
    last_curriculum_update: int,
    frames_since_restart: int,
    curriculum_interval: int,
    prev_episode_successes: Tensor,
    success_tolerance: float,
    initial_tolerance: float,
    target_tolerance: float,
    tolerance_curriculum_increment: float,
) -> Tuple[float, int]:
    """
    Returns: new tolerance, new last_curriculum_update
    """
    if frames_since_restart - last_curriculum_update < curriculum_interval:
        return success_tolerance, last_curriculum_update

    mean_successes_per_episode = prev_episode_successes.mean()    
    if mean_successes_per_episode < 0.5:
        # this policy is not good enough with the previous tolerance value, keep training for now...
        # print(f"Prev episode successes: {mean_successes_per_episode}, same success tolerance: {success_tolerance}")
        return success_tolerance, last_curriculum_update
    
    # decrease the tolerance now
    success_tolerance *= tolerance_curriculum_increment
    success_tolerance = min(success_tolerance, initial_tolerance)
    success_tolerance = max(success_tolerance, target_tolerance)

    print(f"Prev episode successes: {mean_successes_per_episode}, success tolerance: {success_tolerance}")

    last_curriculum_update = frames_since_restart
    return success_tolerance, last_curriculum_update


def interp_0_1(x_curr: float, x_initial: float, x_target: float) -> float:
    """
    Outputs 1 when x_curr == x_target (curriculum completed)
    Outputs 0 when x_curr == x_initial (just started training)
    Interpolates value in between.
    """
    span = x_initial - x_target
    return (x_initial - x_curr) / span


def tolerance_successes_objective(
    success_tolerance: float, initial_tolerance: float, target_tolerance: float, successes: Tensor
) -> Tensor:
    """
    Objective for the PBT. This basically prioritizes tolerance over everything else when we
    execute the curriculum, after that it's just #successes.
    """
    # this grows from 0 to 1 as we reach the target tolerance
    if initial_tolerance > target_tolerance:
        # makeshift unit tests:
        eps = 1e-5
        assert abs(interp_0_1(initial_tolerance, initial_tolerance, target_tolerance)) < eps
        assert abs(interp_0_1(target_tolerance, initial_tolerance, target_tolerance) - 1.0) < eps
        mid_tolerance = (initial_tolerance + target_tolerance) / 2
        assert abs(interp_0_1(mid_tolerance, initial_tolerance, target_tolerance) - 0.5) < eps

        tolerance_objective = interp_0_1(success_tolerance, initial_tolerance, target_tolerance)
    else:
        tolerance_objective = 1.0

    if success_tolerance > target_tolerance:
        # add succeses with a small coefficient to differentiate between policies at the beginning of training
        # increment in tolerance improvement should always give higher value than higher successes with the
        # previous tolerance, that's why this coefficient is very small
        true_objective = (successes * 0.01) + tolerance_objective
    else:
        # basically just the successes + tolerance objective so that true_objective never decreases when we cross
        # the threshold
        true_objective = successes + tolerance_objective

    return true_objective
