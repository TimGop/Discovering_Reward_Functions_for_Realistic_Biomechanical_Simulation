import torch
from typing import Tuple, Dict


def custom_reward_fn_with_video(observation: torch.Tensor, action: torch.Tensor, next_observation: torch.Tensor,
                                terminated: torch.Tensor, truncated: torch.Tensor) \
                                -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Improved reward for humanoid: emphasize fast, stable, human‑like running.

    Design changes based on feedback:
    - r_speed was extremely large and dominated; keep strong but bounded.
    - r_upright was almost saturated (~995/996); reduce its impact via rescaling.
    - r_height was numerically ~constant (~3e‑8); previous scaling was ineffective -> rewrite.
    - r_joint_style, r_damping, r_torque were large negative; rebalance to avoid over‑penalizing.
    - Fitness is OK but can be improved; we make tradeoffs clearer and scales more moderate.
    """

    device = observation.device
    dtype = observation.dtype

    # ----------------------------
    # 0. Basic indices (Humanoid default, exclude root x,y)
    # ----------------------------
    # 0: torso z
    # 1..4: torso quaternion (w,x,y,z)
    # 5..21: joint angles
    # 22..(22+22): various velocities (root lin/ang + joint vels)
    # We keep the same assumptions as in previous runs.

    # ----------------------------
    # 1. Forward velocity reward (re‑scaled)
    # ----------------------------
    vx = next_observation[..., 22]

    # Only reward forward motion
    vx_forward = torch.clamp(vx, min=0.0)

    # Target running speed: ~4–6 m/s. We shape into [0, 1.5] roughly.
    temp_speed_lin = torch.tensor(6.0, device=device, dtype=dtype)
    r_speed_lin = torch.clamp(vx_forward / temp_speed_lin, max=1.0)

    # Quadratic bonus for going faster, but bounded in [0, 1]
    temp_speed_quad = torch.tensor(10.0, device=device, dtype=dtype)
    r_speed_quad = (vx_forward ** 2) / (temp_speed_quad ** 2 + vx_forward ** 2)

    # Combined speed term in [0, ~2.5]
    r_speed = r_speed_lin + r_speed_quad

    # ----------------------------
    # 2. Posture / lean (rescaled)
    # ----------------------------
    torso_quat = next_observation[..., 1:5]

    # Target ~15 deg forward pitch about Y-axis: [cos(a/2), 0, sin(a/2), 0]
    target_quat = torch.tensor([0.99, 0.0, 0.13, 0.0], device=device, dtype=dtype)

    quat_err_sq = torch.sum((torso_quat - target_quat) ** 2, dim=-1)
    temp_upright = torch.tensor(5.0, device=device, dtype=dtype)
    r_upright_raw = torch.exp(-quat_err_sq / temp_upright)  # in (0, 1]

    # Recenter so that "typical" values (~0.9–1.0) are not dominating:
    # map [0,1] → roughly [-0.5, 0.5]
    r_upright = r_upright_raw - 0.5

    # ----------------------------
    # 3. Height reward (rewritten)
    # ----------------------------
    # Previous version used huge scale torso_z (~1100) and tiny temp; it saturated near 0.
    # Here assume torso_z is in simulator units around ~1–2 (if not scaled),
    # but our logs show ~800–1200, so we normalize first.
    torso_z = next_observation[..., 0]

    # Normalize torso height using a rough scale factor (based on logs: ~1000)
    height_norm = torso_z / torch.tensor(1000.0, device=device, dtype=dtype)
    target_height = torch.tensor(1.2, device=device, dtype=dtype)  # prefer ~1.2m
    height_err_sq = (height_norm - target_height) ** 2

    temp_height = torch.tensor(0.1, device=device, dtype=dtype)
    r_height_raw = torch.exp(-height_err_sq / temp_height)  # (0,1]

    # Centered version so deviations matter but don't just add a constant bias
    r_height = r_height_raw - 0.5

    # ----------------------------
    # 4. Joint configuration style (slightly weaker)
    # ----------------------------
    # Penalize large joint angles (avoid extreme crouch / twisted limbs).
    joint_angles = next_observation[..., 5:22]
    joint_dev_sq = torch.sum(joint_angles ** 2, dim=-1)

    temp_joint_dev = torch.tensor(50.0, device=device, dtype=dtype)
    r_joint_style = -joint_dev_sq / temp_joint_dev  # moderate penalty (was too strong before)

    # ----------------------------
    # 5. Smoothness via velocity damping (weakened)
    # ----------------------------
    # Penalize squared velocities to reduce jitter, but not so hard that policy
    # prefers being static.
    qvel = next_observation[..., 22:22 + 23]
    vel_sq = torch.sum(qvel ** 2, dim=-1)

    temp_vel = torch.tensor(400.0, device=device, dtype=dtype)  # doubled vs previous
    r_damping = -vel_sq / temp_vel

    # ----------------------------
    # 6. Energy / torque penalty (weakened)
    # ----------------------------
    action_sq = action ** 2
    torque_cost = torch.sum(action_sq, dim=-1)

    temp_torque = torch.tensor(1.0, device=device, dtype=dtype)  # was 0.5; now milder
    r_torque = -torque_cost / temp_torque

    # ----------------------------
    # 7. Alive bonus & termination penalty (unchanged scale)
    # ----------------------------
    alive_bonus = torch.tensor(0.2, device=device, dtype=dtype)
    r_alive = alive_bonus * (1.0 - terminated.to(dtype))

    death_penalty = torch.tensor(-5.0, device=device, dtype=dtype)
    r_terminate = death_penalty * terminated.to(dtype)

    # ----------------------------
    # 8. Combine into final reward (no base + f(base))
    # ----------------------------
    # Rebalanced weights:
    #  - speed is primary
    #  - posture & height give modest shaping
    #  - style & smoothness are regularizers, not dominating
    w_speed = torch.tensor(4.0, device=device, dtype=dtype)
    w_upright = torch.tensor(0.8, device=device, dtype=dtype)
    w_height = torch.tensor(0.6, device=device, dtype=dtype)
    w_joint_style = torch.tensor(0.5, device=device, dtype=dtype)
    w_damping = torch.tensor(0.3, device=device, dtype=dtype)
    w_torque = torch.tensor(0.3, device=device, dtype=dtype)
    w_alive = torch.tensor(1.0, device=device, dtype=dtype)
    w_terminate = torch.tensor(1.0, device=device, dtype=dtype)

    reward = (
        w_speed * r_speed
        + w_upright * r_upright
        + w_height * r_height
        + w_joint_style * r_joint_style
        + w_damping * r_damping
        + w_torque * r_torque
        + w_alive * r_alive
        + w_terminate * r_terminate
    )

    # ----------------------------
    # 9. Info dict for diagnostics
    # ----------------------------
    info: Dict[str, torch.Tensor] = {
        # Speed
        "r_speed": r_speed,
        "r_speed_lin": r_speed_lin,
        "r_speed_quad": r_speed_quad,
        "forward_velocity": vx,

        # Posture / height
        "r_upright_raw": r_upright_raw,
        "r_upright": r_upright,
        "r_height_raw": r_height_raw,
        "r_height": r_height,
        "torso_z": torso_z,
        "height_norm": height_norm,

        # Style & smoothness
        "r_joint_style": r_joint_style,
        "joint_dev_sq": joint_dev_sq,
        "r_damping": r_damping,
        "vel_sq": vel_sq,

        # Energy & survival
        "r_torque": r_torque,
        "torque_cost": torque_cost,
        "r_alive": r_alive,
        "r_terminate": r_terminate,
    }

    return reward, info
