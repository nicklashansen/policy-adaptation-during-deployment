import torch
import os

from arguments import parse_args
from env.wrappers import make_pad_env
from agent.agent import make_agent
import utils
import time
from logger import Logger
from video import VideoRecorder


def evaluate(env, agent, video, num_episodes, L, step):
	"""Evaluate agent"""
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i == 0))
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs)
			obs, reward, done, _ = env.step(action)
			video.record(env)
			episode_reward += reward

		video.save('%d.mp4' % step)
		L.log('eval/episode_reward', episode_reward, step)
	L.dump(step)


def main(args):
	# Initialize environment
	utils.set_seed_everywhere(args.seed)
	env = make_pad_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		mode=args.mode
	)

	utils.make_dir(args.work_dir)
	model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None)

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	replay_buffer = utils.ReplayBuffer(
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		capacity=args.train_steps,
		batch_size=args.batch_size
	)
	cropped_obs_shape = (3*args.frame_stack, 84, 84)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)

	L = Logger(args.work_dir, use_tb=False)
	episode, episode_reward, done = 0, 0, True
	start_time = time.time()
	for step in range(args.train_steps+1):
		if done:
			if step > 0:
				L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				L.dump(step)

			# Evaluate agent periodically
			if step % args.eval_freq == 0:
				print('Evaluating:', args.work_dir)
				L.log('eval/episode', episode, step)
				evaluate(env, agent, video, args.eval_episodes, L, step)
			
			# Save agent periodically
			if step % args.save_freq == 0 and step > 0:
				if args.save_model:
					agent.save(model_dir, step)

			L.log('train/episode_reward', episode_reward, step)

			obs = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1

			L.log('train/episode', episode, step)

		# Sample action for data collection
		if step < args.init_steps:
			action = env.action_space.sample()
		else:
			with utils.eval_mode(agent):
				action = agent.sample_action(obs)

		# Run training update
		if step >= args.init_steps:
			num_updates = args.init_steps if step == args.init_steps else 1
			for _ in range(num_updates):
				agent.update(replay_buffer, L, step)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		replay_buffer.add(obs, action, reward, next_obs, done_bool)
		episode_reward += reward
		obs = next_obs

		episode_step += 1


if __name__ == '__main__':
	args = parse_args()
	main(args)
