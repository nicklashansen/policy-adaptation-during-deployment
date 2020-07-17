import argparse
import numpy as np
import torch
import os


def main(args):
	assert args.parent_work_dir is not None, 'must specify parent_work_dir'

	eval_rewards = []
	pad_rewards = []

	for seed in range(args.num_seeds):
		results = torch.load(os.path.join(args.parent_work_dir, str(seed), f'pad_{args.mode}.pt'))
		eval_reward = np.mean(results['eval_reward'])
		pad_reward = np.mean(results['pad_reward'])
		eval_rewards.append(eval_reward)
		pad_rewards.append(pad_reward)

	eval_rewards_mean = np.mean(eval_rewards)
	eval_rewards_std = np.std(eval_rewards)
	pad_rewards_mean = np.mean(pad_rewards)
	pad_rewards_std = np.std(pad_rewards)

	print(f'{args.mode}\teval {np.round(eval_rewards_mean)} ({np.round(eval_rewards_std)})\tpad {np.round(pad_rewards_mean)} ({np.round(pad_rewards_std)})\t{args.parent_work_dir}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--parent_work_dir', default=None)
	parser.add_argument('--num_seeds', default=10, type=int)
	parser.add_argument('--mode', default='color_hard', type=str)

	main(parser.parse_args())
