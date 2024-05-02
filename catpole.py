import isaacgym
import isaacgymenvs
import torch
import numpy as np

import sys
import gymnasium
sys.modules["gym"] = gymnasium

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
from stable_baselines3 import PPO, DQN, DDPG
from isaacgymenvs.tasks.base.vec_task import VecTask
from typing import Any, Dict, List


class Sb3VecEnvWrapper(VecEnv):

	def __init__(self, env: VecTask):
		self.env = env
		VecEnv.__init__(self, self.env.num_envs, self.env.observation_space, self.env.action_space)
		self._ep_rew_buf = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)
		self._ep_len_buf = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)

	def get_episode_rewards(self) -> List[float]:
		"""Returns the rewards of all the episodes."""
		return self._ep_rew_buf.cpu().tolist()

	def get_episode_lengths(self) -> List[int]:
		"""Returns the number of time-steps of all the episodes."""
		return self._ep_len_buf.cpu().tolist()

	def reset(self) -> VecEnvObs:  # noqa: D102
		obs_dict = self.env.reset()
		# convert data types to numpy depending on backend
		return self._process_obs(obs_dict)

	def step(self, actions: np.ndarray) -> VecEnvStepReturn:  # noqa: D102
		# convert input to numpy array
		actions = np.asarray(actions)
		# convert to tensor
		actions = torch.from_numpy(actions).to(device=self.env.device)
		# record step information
		obs_dict, rew, dones, extras = self.env.step(actions)

		# update episode un-discounted return and length
		self._ep_rew_buf += rew
		self._ep_len_buf += 1
		reset_ids = (dones > 0).nonzero(as_tuple=False)

		# convert data types to numpy depending on backend
		# Note: IsaacEnv uses torch backend (by default).
		obs = self._process_obs(obs_dict)
		rew = rew.cpu().numpy()
		dones = dones.cpu().numpy()
		# convert extra information to list of dicts
		infos = self._process_extras(obs, dones, extras, reset_ids)

		# reset info for terminated environments
		self._ep_rew_buf[reset_ids] = 0
		self._ep_len_buf[reset_ids] = 0

		return obs, rew, dones, infos

	"""
    Unused methods.
    """

	def step_async(self, actions):  # noqa: D102
		self._async_actions = actions

	def step_wait(self):  # noqa: D102
		return self.step(self._async_actions)

	def get_attr(self, attr_name, indices=None):  # noqa: D102
		return self.env.get_attr(attr_name, indices)

	# raise NotImplementedError

	def set_attr(self, attr_name, value, indices=None):  # noqa: D102
		return self.env.set_attr(attr_name, value, indices)

	def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):  # noqa: D102
		return self.env.env_method(method_name, *method_args, indices=indices, **method_kwargs)

	def env_is_wrapped(self, wrapper_class, indices=None):  # noqa: D102
		return self.env.env_is_wrapped(wrapper_class, indices=indices)

	def get_images(self):  # noqa: D102
		return self.env.get_images()

	def close(self) -> None:
		return self.venv.close()

	"""
    Helper functions.
    """

	def _process_obs(self, obs_dict) -> np.ndarray:
		"""Convert observations into NumPy data type."""
		# Sb3 doesn't support asymmetric observation spaces, so we only use "policy"
		# obs = obs_dict["policy"]
		obs = obs_dict["obs"]
		# Note: IsaacEnv uses torch backend (by default).
		# if self.env.sim.backend == "torch":
		if self.env.device == "cuda:0":
			if isinstance(obs, dict):
				for key, value in obs.items():
					obs[key] = value.detach().cpu().numpy()
			else:
				obs = obs.detach().cpu().numpy()
		# elif self.env.sim.backend == "numpy":
		elif self.env.device == "cpu":
			pass
		else:
			raise NotImplementedError(f"Unsupported backend for simulation: {self.env.sim.backend}")
		return obs

	def _process_extras(self, obs, dones, extras, reset_ids) -> List[Dict[str, Any]]:
		"""Convert miscellaneous information into dictionary for each sub-environment."""
		# create empty list of dictionaries to fill
		infos: List[Dict[str, Any]] = [dict.fromkeys(extras.keys()) for _ in range(self.env.num_envs)]
		# fill-in information for each sub-environment
		# Note: This loop becomes slow when number of environments is large.
		for idx in range(self.env.num_envs):
			# fill-in episode monitoring info
			if idx in reset_ids:
				infos[idx]["episode"] = dict()
				infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
				infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
			else:
				infos[idx]["episode"] = None
			# fill-in information from extras
			for key, value in extras.items():
				# 1. remap the key for time-outs for what SB3 expects
				# 2. remap extra episodes information safely
				# 3. for others just store their values
				if key == "time_outs":
					infos[idx]["TimeLimit.truncated"] = bool(value[idx])
				elif key == "episode":
					# only log this data for episodes that are terminated
					if infos[idx]["episode"] is not None:
						for sub_key, sub_value in value.items():
							infos[idx]["episode"][sub_key] = sub_value
				else:
					infos[idx][key] = value[idx]
			# add information about terminal observation separately
			if dones[idx] == 1:
				# extract terminal observations
				if isinstance(obs, dict):
					terminal_obs = dict.fromkeys(obs.keys())
					for key, value in obs.items():
						terminal_obs[key] = value[idx]
				else:
					terminal_obs = obs[idx]
				# add info to dict
				infos[idx]["terminal_observation"] = terminal_obs
			else:
				infos[idx]["terminal_observation"] = None
		# return list of dictionaries
		return infos




num_envs = 512

envs = isaacgymenvs.make(
	seed=0,
	task="Cartpole",
	num_envs=num_envs,
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
	# headless=False,
	# multi_gpu=False,
	# virtual_screen_capture=False,
	# force_render=False,
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)

sb3Env = Sb3VecEnvWrapper(envs)

model = PPO("MlpPolicy", sb3Env, n_steps=500, batch_size=512, ent_coef=0.01, tensorboard_log="./a2c_cartpole_tensorboard/", verbose=0)
model.learn(total_timesteps=10_000_000, tb_log_name="first_run", progress_bar=True)
model.save("ppo_cartpole")


# obs = envs.reset()
# for _ in range(20):
# 	random_actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
# 	envs.step(random_actions)

print("Model saved---------------------")

obs = sb3Env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = sb3Env.step(action)
    # if dones:
    #   obs = sb3Env.reset()
    # sb3Env.render("human")