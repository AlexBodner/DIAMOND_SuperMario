<<<<<<< HEAD
from typing import Tuple, Union

import numpy as np
from PIL import Image

from csgo.action_processing import CSGOAction
from .dataset_env import DatasetEnv
from .play_env import PlayEnv
from IPython.display import display, clear_output
from ipywidgets import widgets
from IPython.display import clear_output
from IPython.display import display
import asyncio
import nest_asyncio


steering_value = 0.0
stop_task = False

class ColabGame:
	def __init__(
		self,
		play_env: Union[PlayEnv, DatasetEnv],
		size: Tuple[int, int],
		mouse_multiplier: int,
		fps: int,
		verbose: bool,
	) -> None:
		self.env = play_env
		self.height, self.width = size
		self.mouse_multiplier = mouse_multiplier
		self.fps = fps
		self.verbose = verbose

		print("\nControls:\n")
		print(" m  : switch control (human/replay)") # Not for main as Game can use either PlayEnv or DatasetEnv
		print(" .  : pause/unpause")
		print(" e  : step-by-step (when paused)")
		print(" ⏎  : reset env")
		print("Esc : quit")
		print("\n")

	def on_slider_change(self, change):
		global steering_value
		steering_value = change['new']  # Update the global variable

	def stop_monitoring(self, change):
		global stop_task
		stop_task = True  # Set the flag to stop the task

	def run(self) -> None:
		global stop_task
		nest_asyncio.apply()

		original_slider = widgets.FloatSlider(min=-1, max=1, step=0.1, value=0, description="Steering")
		output = widgets.Output()
		original_slider.observe(self.on_slider_change, names='value')
		display(original_slider)

		stop_button = widgets.Button(description="Stop Task")
		stop_button.on_click(self.stop_monitoring)

		# Display the output widget and stop button
		display(output, stop_button)
		stop_task = False
		asyncio.create_task(self.async_run())

	async def async_run(self) -> None:
		global steering_value
		async def draw_obs(obs, obs_low_res=None):
			assert obs.ndim == 4 and obs.size(0) == 1
			clear_output(wait=True)
			img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
			display(img)
			#plt.imshow(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy(), interpolation='nearest')
			#plt.show()
			#display(slider)
			print(steering_value)
			

		async def reset():
			nonlocal obs, info, do_reset, ep_return, ep_length, keys_pressed, l_click, r_click
			global steering_value
			obs, info = await asyncio.to_thread(self.env.reset)
			do_reset = False
			ep_return = 0
			ep_length = 0
			keys_pressed = []
			steering_value = 0.0
			l_click = r_click = False

		print(steering_value)
		await reset()
		
		obs, info, do_reset, ep_return, ep_length, keys_pressed, l_click, r_click = (None,) * 8
		

		while not stop_task:
			if do_reset:
				await reset()
			do_wait = False
			await asyncio.sleep(0.01)
			csgo_action = CSGOAction(steering_value)
			next_obs, rew, end, trunc, info = await asyncio.to_thread(self.env.step, csgo_action)

			await draw_obs(next_obs)
			if end or trunc:
				await reset()

			else:
=======
from typing import Tuple, Union

import numpy as np
from PIL import Image

from csgo.action_processing import CSGOAction
from .dataset_env import DatasetEnv
from .play_env import PlayEnv
from IPython.display import display, clear_output
from ipywidgets import widgets
from IPython.display import clear_output
from IPython.display import display
import asyncio
import nest_asyncio


steering_value = 0.0
stop_task = False

class ColabGame:
	def __init__(
		self,
		play_env: Union[PlayEnv, DatasetEnv],
		size: Tuple[int, int],
		mouse_multiplier: int,
		fps: int,
		verbose: bool,
	) -> None:
		self.env = play_env
		self.height, self.width = size
		self.mouse_multiplier = mouse_multiplier
		self.fps = fps
		self.verbose = verbose

		print("\nControls:\n")
		print(" m  : switch control (human/replay)") # Not for main as Game can use either PlayEnv or DatasetEnv
		print(" .  : pause/unpause")
		print(" e  : step-by-step (when paused)")
		print(" ⏎  : reset env")
		print("Esc : quit")
		print("\n")

	def on_slider_change(self, change):
		global steering_value
		steering_value = change['new']  # Update the global variable

	def stop_monitoring(self, change):
		global stop_task
		stop_task = True  # Set the flag to stop the task

	def run(self) -> None:
		global stop_task
		nest_asyncio.apply()

		original_slider = widgets.FloatSlider(min=-1, max=1, step=0.1, value=0, description="Steering")
		output = widgets.Output()
		original_slider.observe(self.on_slider_change, names='value')
		display(original_slider)

		stop_button = widgets.Button(description="Stop Task")
		stop_button.on_click(self.stop_monitoring)

		# Display the output widget and stop button
		display(output, stop_button)
		stop_task = False
		asyncio.create_task(self.async_run())

	async def async_run(self) -> None:
		global steering_value
		async def draw_obs(obs, obs_low_res=None):
			assert obs.ndim == 4 and obs.size(0) == 1
			clear_output(wait=True)
			img = Image.fromarray(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
			display(img)
			#plt.imshow(obs[0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy(), interpolation='nearest')
			#plt.show()
			#display(slider)
			print(steering_value)
			

		async def reset():
			nonlocal obs, info, do_reset, ep_return, ep_length, keys_pressed, l_click, r_click
			global steering_value
			obs, info = await asyncio.to_thread(self.env.reset)
			do_reset = False
			ep_return = 0
			ep_length = 0
			keys_pressed = []
			steering_value = 0.0
			l_click = r_click = False

		print(steering_value)
		await reset()
		
		obs, info, do_reset, ep_return, ep_length, keys_pressed, l_click, r_click = (None,) * 8
		

		while not stop_task:
			if do_reset:
				await reset()
			do_wait = False
			await asyncio.sleep(0.01)
			csgo_action = CSGOAction(steering_value)
			next_obs, rew, end, trunc, info = await asyncio.to_thread(self.env.step, csgo_action)

			await draw_obs(next_obs)
			if end or trunc:
				await reset()

			else:
>>>>>>> origin/master
				obs = next_obs