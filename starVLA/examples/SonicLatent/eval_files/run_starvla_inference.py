"""
starVLA inference runner for Sonic latent actions.

This script mirrors the deployment flow of `gear_sonic/scripts/run_vla_inference.py`
but talks to the starVLA WebSocket policy server instead of Isaac-GR00T's
PolicyClient server.
"""

from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time

import cv2 as cv
import numpy as np
import tyro
import zmq

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.features_sonic_vla import get_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


@dataclass
class InferenceConfig:
    ckpt_path: str
    """Trained starVLA checkpoint path, used for loading normalization stats."""

    host: str = "127.0.0.1"
    """starVLA WebSocket policy server host."""

    port: int = 10093
    """starVLA WebSocket policy server port."""

    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 0
    """Deprecated compatibility flag. Actual action chunk size is read from the checkpoint."""

    rate: float = 4
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    camera_host: str = "localhost"
    camera_port: int = 5555
    state_zmq_host: str = "localhost"
    state_zmq_port: int = 5557
    action_zmq_host: str = "localhost"
    action_zmq_port: int = 5556
    keyboard_zmq_host: str = "localhost"
    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    prompt: str = "pick up the cylinder and throw it into the trash bin"
    image_size: tuple[int, int] = (224, 224)
    use_ddim: bool = False
    num_ddim_steps: int = 10
    verbose_timing: bool = False
    log_action_stats: bool = False
    latency_compensation: bool = False


def print_green(x):
    print(f"\033[92m{x}\033[0m")


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def _sleep_remaining(t_start: float, loop_period: float):
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


class StarVLAPolicyAdapter:
    """Thin client-side adapter around the starVLA WebSocket server."""

    def __init__(self, ckpt_path: str, host: str, port: int, image_size: tuple[int, int]):
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.ckpt_path = Path(ckpt_path)
        self.model_config, self.norm_stats = read_mode_config(self.ckpt_path)
        self.action_stats = self._get_action_stats()
        self.state_stats = self._get_state_stats()
        self.action_chunk_size = self.model_config["framework"]["action_model"]["future_action_window_size"] + 1
        self.image_size = tuple(image_size)

    def _get_action_stats(self) -> dict:
        assert len(self.norm_stats) >= 1, "No normalization stats found in checkpoint config."
        dataset_key = next(iter(self.norm_stats.keys()))
        return self.norm_stats[dataset_key]["action"]

    def _get_state_stats(self) -> dict:
        assert len(self.norm_stats) >= 1, "No normalization stats found in checkpoint config."
        dataset_key = next(iter(self.norm_stats.keys()))
        return self.norm_stats[dataset_key]["state"]

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        state_min = np.asarray(self.state_stats["min"], dtype=np.float32)
        state_max = np.asarray(self.state_stats["max"], dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        mask = state_min != state_max
        normalized = np.zeros_like(state, dtype=np.float32)
        normalized[..., mask] = 2.0 * (state[..., mask] - state_min[mask]) / (
            state_max[mask] - state_min[mask]
        ) - 1.0
        return normalized

    def _unnormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        mask = self.action_stats.get("mask", np.ones_like(self.action_stats["min"], dtype=bool))
        action_high = np.asarray(self.action_stats["max"], dtype=np.float32)
        action_low = np.asarray(self.action_stats["min"], dtype=np.float32)
        normalized_actions = np.clip(normalized_actions, -1, 1)
        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        ).astype(np.float32)

    def predict_action(
        self,
        image: np.ndarray,
        state: np.ndarray,
        language_prompt: str,
        use_ddim: bool,
        num_ddim_steps: int,
    ) -> dict:
        resized = cv.resize(image, self.image_size, interpolation=cv.INTER_AREA)
        normalized_state = self._normalize_state(state)
        example = {
            "image": [resized],
            "lang": language_prompt,
            "state": normalized_state[np.newaxis, :].astype(np.float32),
        }
        response = self.client.predict_action(
            {
                "examples": [example],
                "do_sample": False,
                "use_ddim": use_ddim,
                "num_ddim_steps": num_ddim_steps,
            }
        )
        normalized_actions = response["data"]["normalized_actions"][0]
        actions = self._unnormalize_actions(normalized_actions)
        return {
            "motion_token": actions[:, :64],
            "left_hand_joints": actions[:, 64:71],
            "right_hand_joints": actions[:, 71:78],
        }


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
):
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    image = camera_msg["images"]["ego_view"]

    left_hand_q = np.asarray(state_msg["left_hand_q"], dtype=np.float32).copy()
    right_hand_q = np.asarray(state_msg["right_hand_q"], dtype=np.float32).copy()
    body_q = np.asarray(state_msg["body_q"], dtype=np.float32)

    # Copy index finger data to middle finger (hardware coupling)
    left_hand_q[5] = left_hand_q[3]
    left_hand_q[6] = left_hand_q[4]

    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat).astype(np.float32)

    whole_q = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=body_q,
        left_hand_actuated_joint_values=left_hand_q,
        right_hand_actuated_joint_values=right_hand_q,
    ).astype(np.float32)

    state = np.concatenate([whole_q, projected_gravity], axis=0)
    assert state.shape[0] == 46, f"Expected state dim 46, got {state.shape[0]}"

    return {
        "image": image,
        "state": state,
        "language_prompt": language_prompt,
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }


def _format_range(name: str, value: np.ndarray) -> str:
    value = np.asarray(value, dtype=np.float32)
    return (
        f"{name}: shape={value.shape}, "
        f"min={value.min():.4f}, max={value.max():.4f}, "
        f"first={np.array2string(value[0], precision=3, suppress_small=True)}, "
        f"last={np.array2string(value[-1], precision=3, suppress_small=True)}"
    )


def _format_action_samples(name: str, value: np.ndarray) -> str:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        return f"{name}[0]={np.array2string(value, precision=3, suppress_small=True)}"
    indices = sorted(set([0, value.shape[0] // 2, value.shape[0] - 1]))
    parts = [
        f"{name}[{idx}]={np.array2string(value[idx], precision=3, suppress_small=True)}"
        for idx in indices
    ]
    return "; ".join(parts)


def run_policy_inference_and_process(
    policy: StarVLAPolicyAdapter,
    observation: dict,
    use_ddim: bool,
    num_ddim_steps: int,
    log_action_stats: bool = False,
):
    try:
        processed_action = policy.predict_action(
            image=observation["image"],
            state=observation["state"],
            language_prompt=observation["language_prompt"],
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
        )
        if np.abs(processed_action["motion_token"]).max() > 1.25:
            print(
                f"[Warning] motion_token max ({np.abs(processed_action['motion_token']).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None
        if log_action_stats:
            print_green(_format_range("motion_token", processed_action["motion_token"]))
            print_green(_format_range("left_hand", processed_action["left_hand_joints"]))
            print_green(_format_range("right_hand", processed_action["right_hand_joints"]))
            print_green(_format_action_samples("left_hand", processed_action["left_hand_joints"]))
            print_green(_format_action_samples("right_hand", processed_action["right_hand_joints"]))
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    while not stop_event.is_set():
        try:
            try:
                inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn()
                if observation is None:
                    continue
                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)
                if processed_action is not None:
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time))
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


def main(config: InferenceConfig):
    pause_loop = True
    robot_model = get_g1_robot_model(waist_location="lower_and_upper_body")
    policy = StarVLAPolicyAdapter(
        ckpt_path=config.ckpt_path,
        host=config.host,
        port=config.port,
        image_size=config.image_size,
    )
    print(f"Connected to starVLA policy server at {config.host}:{config.port}")
    print_green(f"Action chunk size from checkpoint: {policy.action_chunk_size}")

    state_subscriber = ZMQStateSubscriber(host=config.state_zmq_host, port=config.state_zmq_port)
    camera_subscriber = ComposedCameraClientSensor(server_ip=config.camera_host, port=config.camera_port)

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}")

    keyboard_listener = ZMQKeyboardSubscriber(port=config.keyboard_zmq_port, host=config.keyboard_zmq_host)
    telemetry = Telemetry(window_size=100)

    loop_period = 1.0 / config.action_publish_rate
    cpp_loop_running = False
    cpp_mode = "OFF"
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate
    zmq_frame_counter = 0
    language_prompt_ref: list[str] = [config.prompt]
    prompt_prefix = "prompt:"

    def publish_initial_pose():
        left_hand = _compute_closed_hand_joints("L") if initial_pose_left_hand_closed else np.zeros(7, dtype=np.float32)
        right_hand = _compute_closed_hand_joints("R") if initial_pose_right_hand_closed else np.zeros(7, dtype=np.float32)
        zmq_message = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)

    def send_cpp_control_command(start: bool, planner: bool = False):
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            cpp_loop_running = start
            cpp_mode = "PLANNER" if (start and planner) else ("POSE" if start else "OFF")
            return True
        except Exception as e:
            print(f"Warning: Failed to send control command: {e}")
            return False

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time, zmq_frame_counter

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(prompt_prefix):
            new_prompt = key[len(prompt_prefix):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            return

        if key == "i":
            zmq_frame_counter = 0
            publish_initial_pose()
            cached_action_chunk = None
            action_chunk_index = 0
            if cpp_loop_running and cpp_mode == "PLANNER":
                send_cpp_control_command(start=True, planner=False)
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
        elif key == "k":
            if cpp_loop_running:
                send_cpp_control_command(start=False, planner=(cpp_mode == "PLANNER"))
            else:
                send_cpp_control_command(start=True, planner=True)
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
                log_errors=True,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=policy,
                observation=obs,
                use_ddim=config.use_ddim,
                num_ddim_steps=config.num_ddim_steps,
                log_action_stats=config.log_action_stats,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            try:
                processed_action, inference_start_time = result_queue.get_nowait()
                inference_delay = time.monotonic() - inference_start_time
                action_chunk_index = (
                    calculate_latency_compensated_index(
                        inference_delay, config.action_publish_rate, policy.action_chunk_size
                    )
                    if config.latency_compensation
                    else 0
                )
                cached_action_chunk = processed_action
                last_inference_time = time.monotonic()
                print_green(
                    f'New action chunk (prompt: "{language_prompt_ref[0]}", latency: {inference_delay:.3f}s)'
                )
            except queue.Empty:
                pass

            should_start = should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=inference_busy_event.is_set(),
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            )

            if should_start:
                try:
                    inference_queue.put_nowait(None)
                except queue.Full:
                    pass

            if pause_loop:
                time.sleep(0.2)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    _sleep_remaining(t_start, loop_period)
                    continue

                motion_token = np.asarray(cached_action_chunk["motion_token"], dtype=np.float32)
                left_hand_joints = np.asarray(cached_action_chunk["left_hand_joints"], dtype=np.float32)
                right_hand_joints = np.asarray(cached_action_chunk["right_hand_joints"], dtype=np.float32)

                horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                current_idx = min(action_chunk_index, horizon - 1)

                if motion_token.ndim == 2:
                    motion_token = motion_token[current_idx]
                if left_hand_joints.ndim == 2:
                    left_hand_joints = left_hand_joints[current_idx]
                if right_hand_joints.ndim == 2:
                    right_hand_joints = right_hand_joints[current_idx]

                frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                zmq_frame_counter += 1

                zmq_message = pack_latent_action_message(
                    motion_token,
                    frame_index,
                    left_hand_joints=left_hand_joints,
                    right_hand_joints=right_hand_joints,
                )
                zmq_socket.send(zmq_message)
                action_chunk_index = min(action_chunk_index + 1, policy.action_chunk_size - 1)

            if config.verbose_timing and (time.monotonic() - t_start) > 0:
                telemetry.log_timing_info(context="starVLA Inference Loop", threshold=0.0)

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("starVLA inference loop terminated by user")
    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        keyboard_listener.close()
        policy.client.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
