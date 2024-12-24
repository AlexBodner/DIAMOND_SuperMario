import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from agent import Agent
from envs import WorldModelEnv
from game import Game, PlayEnv, ColabGame
<<<<<<< HEAD
=======
from utils import count_parameters
>>>>>>> origin/master

OmegaConf.clear_resolvers()
OmegaConf.register_new_resolver("eval", eval)




def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--record", action="store_true", help="Record episodes in PlayEnv.")
	parser.add_argument("--store-denoising-trajectory", action="store_true", help="Save denoising steps in info.")
	parser.add_argument("--store-original-obs", action="store_true", help="Save original obs (pre resizing) in info.")
	parser.add_argument("--mouse-multiplier", type=int, default=10, help="Multiplication factor for the mouse movement.")
	parser.add_argument("--size-multiplier", type=int, default=2, help="Multiplication factor for the screen size.")
	parser.add_argument("--compile", action="store_true", help="Turn on model compilation.")
	parser.add_argument("--colab", action="store_true", help="Run the model on COLAB, without using PyGame.")
	parser.add_argument("--fps", type=int, default=15, help="Frame rate.")
	parser.add_argument("--no-header", action="store_true")
	parser.add_argument(
		"--model",
		type=Path,
		help="If you want to use a custom model, specify the directory.",
	)
	parser.add_argument("--save-frames", type=str, help="Path to save frames captured during gameplay.")

	return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
	if not args.record and (args.store_denoising_trajectory or args.store_original_obs):
		print("Warning: not in recording mode, ignoring --store* options")
	return True


def prepare_play_mode(cfg: DictConfig, args: argparse.Namespace) -> PlayEnv:
	#if args.model == "" or args.model == None:
	#	path_hf =  Path(snapshot_download(repo_id="DereWah/diamond-mariokart64", allow_patterns="csgo/*"))
	#else:
	path_hf = args.model		
	# If you're running with a model that you already downloaded. simply change this path above to match the folder in which you put the model
	path_ckpt ="checkpoints/human_data_epoch50.pt"# path_hf / "csgo/model/csgo.pt"
	spawn_dir = path_hf / "csgo/spawn"

	# Override config
	cfg.agent = OmegaConf.load("config/agent/csgo.yaml")
	cfg.env = OmegaConf.load("config/env/csgo.yaml")

	if torch.cuda.is_available():
		device = torch.device("cuda:0")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")

	assert cfg.env.train.id == "csgo"
	num_actions = cfg.env.num_actions

	# Models
	agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
	agent.load(path_ckpt)
	
	# World model environment
	sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
	if agent.upsampler is not None:
		sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)
	wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
	wm_env = WorldModelEnv(agent.denoiser, agent.upsampler, agent.rew_end_model, spawn_dir, 1, sl, wm_env_cfg, return_denoising_trajectory=True)
<<<<<<< HEAD
	
=======
	for name in ["denoiser", "upsampler"]:
			print(f"{count_parameters(getattr(agent, name))} parameters in {name}")
>>>>>>> origin/master
	if device.type == "cuda" and args.compile:
		print("Compiling models...")
		wm_env.predict_next_obs = torch.compile(wm_env.predict_next_obs, mode="reduce-overhead")
		wm_env.upsample_next_obs = torch.compile(wm_env.upsample_next_obs, mode="reduce-overhead")

	play_env = PlayEnv(
		agent,
		wm_env,
		args.record,
		args.store_denoising_trajectory,
		args.store_original_obs,
	)

	return play_env
import os
from PIL import Image

def save_frame(obs, frame_count, save_path):
    """
    Save a single frame to the specified path.
    Args:
        obs: Observation frame to save.
        frame_count: The current frame count (used for naming the file).
        save_path: Directory to save frames.
    """
    os.makedirs(save_path, exist_ok=True)
    frame_image = Image.fromarray(
        obs.add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    )
    frame_image.save(os.path.join(save_path, f"frame_{frame_count:04d}.png"))

@torch.no_grad()
def main():
	args = parse_args()
	ok = check_args(args)
	if not ok:
		return

	with initialize(version_base="1.3", config_path="../config"):
		cfg = compose(config_name="trainer")
	print(cfg)
	# window size
	w, h = (cfg.env.train.size,) * 2 if isinstance(cfg.env.train.size, int) else cfg.env.train.size
	size_h, size_w = h * args.size_multiplier, w * args.size_multiplier
	env = prepare_play_mode(cfg, args)
	save_path = args.save_frames if args.save_frames else None

	if args.colab:
		colab_game = ColabGame(env, (size_h, size_w), args.mouse_multiplier, fps=args.fps, verbose=not args.no_header)
		colab_game.run()
	else:
		game = Game(env, (size_h, size_w), args.mouse_multiplier, fps=args.fps, verbose=not args.no_header,save_frames_dir = save_path)
		game.run()


if __name__ == "__main__":
	
	main()
