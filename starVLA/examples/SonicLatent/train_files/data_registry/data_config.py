"""Sonic latent-action benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class SonicLatentDataConfig:
    def __init__(self, action_horizon: int = 40):
        self.action_horizon = int(action_horizon)

    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.left_hand",
        "state.right_arm",
        "state.right_hand",
        "state.projected_gravity",
    ]
    action_keys = [
        "action.motion_token",
        "action.left_hand_joints",
        "action.right_hand_joints",
    ]
    language_keys = ["annotation.human.task_description"]

    observation_indices = [0]
    state_indices = [0]

    def modality_config(self):
        action_indices = list(range(self.action_horizon))
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_leg": "min_max",
                    "state.right_leg": "min_max",
                    "state.waist": "min_max",
                    "state.left_arm": "min_max",
                    "state.left_hand": "min_max",
                    "state.right_arm": "min_max",
                    "state.right_hand": "min_max",
                    "state.projected_gravity": "min_max",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.motion_token": "min_max",
                    "action.left_hand_joints": "min_max",
                    "action.right_hand_joints": "min_max",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "sonic_latent_humanoid": SonicLatentDataConfig(),
}


ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "sonic_latent_humanoid": EmbodimentTag.NEW_EMBODIMENT,
}


DATASET_NAMED_MIXTURES = {
    "sonic_merged_dataset": [
        ("merged_dataset_001", 1.0, "sonic_latent_humanoid"),
    ],
    "sonic_merged_dataset_001": [
        ("merged_dataset_001", 1.0, "sonic_latent_humanoid"),
    ],

}
