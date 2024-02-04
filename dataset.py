# robomimic dataset
import os
import json
import h5py
import numpy as np
import robosuite
from robosuite.controllers import load_controller_config


class RobomimicDataset:
    def __init__(self, path, cameras=['agentview', 'robot0_eye_in_hand'], num_episode=None):



        self.data = []
        datafile = h5py.File(path)
        num_episode: int = len(list(datafile["data"].keys())) if num_episode is None else num_episode  # type: ignore
        print(f"Raw Dataset size (#episode): {num_episode}")
        
        self.idx2entry = []  # store idx -> (episode_idx, timestep_idx)
        self.data_pairs = []
        self.action_pairs = []

        self.cameras = cameras
        self.obs = []
        self.actions = []
        self.idx2entry = []
        for episode_id in range(num_episode):
            print(episode_id)

            episode = datafile[f"data/demo_{episode_id}"]
            episode_obs = {}
            episode_action = np.array(episode["actions"], dtype=np.float32)
            for camera in cameras:

                episode_obs[camera] = np.array(episode[f"obs/{camera}_image"], dtype=np.float32)/255

            self.obs.append(episode_obs)
            self.actions.append(episode_action)

            for e in range(episode_action.shape[0]):
                self.idx2entry.append((episode_id, e))

        datafile.close()

        config_path = os.path.join(os.path.dirname(path), "env_cfg.json")
        self.env_config = json.load(open(config_path, "r"))
        # Load the default controller configuration
        controller_configs = load_controller_config(default_controller="OSC_POSE")
        controller_configs['control_delta'] = self.env_config["env_kwargs"]["controller_configs"]["control_delta"]
        self.env = robosuite.make(
            env_name=self.env_config["env_name"],
            robots=self.env_config["env_kwargs"]["robots"],
            controller_configs=controller_configs,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            reward_shaping=False,
            camera_names=cameras,
            camera_heights=96,
            camera_widths=96,
            horizon=300,
        )

    def __len__(self):
        return len(self.idx2entry)

    def __getitem__(self, idx):
        episode_id, episode_entry_id = self.idx2entry[idx]
        obs = {}
        for camera in self.cameras:
            obs[camera] = self.obs[episode_id][camera][episode_entry_id]
        action = self.actions[episode_id][episode_entry_id]
        
        return obs, action

if __name__ == '__main__':
    dataset = RobomimicDataset(
        path='data/robomimic/square/processed_data96.hdf5',
        num_episode=1
    )

    for obs, action in dataset:
        breakpoint()