import torch
import os
import pandas as pd
import numpy as np
import gym
from algorithms.rl_utils import make_obs_grad_grid
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def evaluate(env, agent, video, num_episodes, L, step, eval_mode='normal'):
	episode_rewards = []
	for i in range(num_episodes):
		obs = env.reset()
		#video.init(enabled=(i==0))
		video.init()
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs, eval_mode)
			obs, reward, done, _ = env.step(action)
			
			frame = np.moveaxis(obs[-3:],0,2)
			video.record(frame)

			episode_reward += reward

		video_path = 'step{}_test{}_{}.mp4'.format(step, i, eval_mode)
		video.save(video_path)
		    
		episode_rewards.append(episode_reward)
	
	if L is not None:
		L.log('eval/episode_reward_{}'.format(eval_mode), np.mean(episode_rewards), step)

	return np.mean(episode_rewards)

def make_envs(args):
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='train'
	)
	color_hard_test_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='color_hard',
		intensity=args.distracting_cs_intensity
	)
	video_easy_test_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='video_easy',
		intensity=args.distracting_cs_intensity
	)
	video_hard_test_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='video_hard',
		intensity=args.distracting_cs_intensity
	)
	return env, color_hard_test_env, video_easy_test_env, video_hard_test_env

def main(args):
	START = time.time()
	# Set seed
	utils.set_seed_everywhere(args.seed)

	# Initialize environments
	gym.logger.set_level(40)
	env, color_hard_test_env, video_easy_test_env, video_hard_test_env = make_envs(args)

	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, 'time_'+datetime.now().isoformat() + '_seed_'+str(args.seed))
	print("Working directory:", work_dir)
	assert not os.path.exists(os.path.join(work_dir, "train.log")), "specified working directory already exists"
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, "model"))
	video_dir = utils.make_dir(os.path.join(work_dir, "video"))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
	utils.write_info(args, os.path.join(work_dir, "info.log"))

	train_cols = ['episode', 'episode_reward', 'critic_loss', 'recon_loss', 'actor_loss', 'alpha_loss', 'alpha_value', 'duration', 'step', 'env_step']
	train_csv = pd.DataFrame(columns = train_cols)
	train_csv_fname = os.path.join(work_dir, 'train.csv')

	eval_cols = ['episode', 'episode_reward_normal', 'episode_reward_color_hard', 'episode_reward_video_easy', 'episode_reward_video_hard', 'step', 'env_step', 'eval_duration']
	eval_csv = pd.DataFrame(columns = eval_cols)
	eval_csv_fname = os.path.join(work_dir, 'eval.csv')
    
    # Prepare agent
    #assert torch.cuda.is_available(), "must have cuda enabled"
	replay_buffer = utils.ReplayBuffer(
	    obs_shape=env.observation_space.shape,
	    action_shape=env.action_space.shape,
	    capacity=args.train_steps,
	    batch_size=args.batch_size,
	)
	cropped_obs_shape = (
	    3 * args.frame_stack,
	    args.image_crop_size,
	    args.image_crop_size,
	)
	print("Observations:", env.observation_space.shape)
	print("Cropped observations:", cropped_obs_shape)
	agent = make_agent(
	    obs_shape=cropped_obs_shape, action_shape=env.action_space.shape, args=args
	)
	agent.work_dir = work_dir
	
	print('number of encoder parameters {}'.format(utils.count_parameters(agent.critic.encoder)))
	print('number of shared_CNN parameters {}'.format(utils.count_parameters(agent.critic.encoder.shared_cnn)))
	print('number of linear projection parameters {}'.format(utils.count_parameters(agent.critic.encoder.projection)))
	print('number of Q1,Q2 parameters {}'.format(utils.count_parameters(agent.critic.Q1)))
	print('number of actor head parameters {}'.format(utils.count_parameters(agent.actor.mlp)))

	start_step, episode, episode_step, episode_reward, done = 0, 0, 0, 0, True
	L = Logger(work_dir)
	start_time = time.time()
	for step in range(start_step, args.train_steps + 1):
		if done:
			if step > start_step:
				# log training results
				L.log('train/episode', episode, step)
				L.log('train/episode_reward', episode_reward, step)

				duration = time.time() - start_time
				L.log('train/duration', duration, step)
					
				L.dump(step)
					
				# log training results to 'train.csv'
				if len(critic_losses) > 0:
					train_csv.loc[len(train_csv.index)] = [episode, episode_reward, np.mean(critic_losses), np.mean(recon_losses), np.mean(actor_losses), np.mean(alpha_losses), np.mean(alpha_values), duration, step, step*args.action_repeat]
				else:
					train_csv.loc[len(train_csv.index)] = [episode, episode_reward, 0, 0, 0, 0, 0, duration, step, step*args.action_repeat]
				train_csv.to_csv(train_csv_fname)

			# Evaluate agent periodically
			if step % args.eval_freq == 0:
				eval_start = time.time()
				L.log('eval/episode', episode, step)
				eval_reward_normal = evaluate(env, agent, video, args.eval_episodes, L, step, 'normal')
				#eval_reward_color_hard, eval_reward_video_easy, eval_reward_video_hard = 0, 0, 0
				eval_reward_color_hard = evaluate(color_hard_test_env, agent, video, args.eval_episodes, L, step, 'color_hard')
				eval_reward_video_easy = evaluate(video_easy_test_env, agent, video, args.eval_episodes, L, step, 'video_easy')
				eval_reward_video_hard = evaluate(video_hard_test_env, agent, video, args.eval_episodes, L, step, 'video_hard')
				eval_end = time.time()
				L.dump(step)
			
				# log evaluation results to 'eval.csv'
				eval_csv.loc[len(eval_csv.index)] = [episode, eval_reward_normal, eval_reward_color_hard, eval_reward_video_easy, eval_reward_video_hard, step, step*args.action_repeat, eval_end-eval_start]
				eval_csv.to_csv(eval_csv_fname)

			# Save agent periodically
			if step > start_step and step % args.save_freq == 0:
				torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

			# reset
			critic_losses, recon_losses, actor_losses, alpha_losses, alpha_values = [], [], [], [], []
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1
			start_time = time.time()				

			# reset the environment
			obs = env.reset()

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
				critic_loss, recon_loss, actor_loss, alpha_loss, alpha_value = agent.update(replay_buffer, L, step)
				critic_losses.append(critic_loss)
				if recon_loss != None:
					recon_losses.append(recon_loss)
				if actor_loss != None:
					actor_losses.append(actor_loss)
					alpha_losses.append(alpha_loss)
					alpha_values.append(alpha_value)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		replay_buffer.add(obs, action, reward, next_obs, done_bool)
		episode_reward += reward
		obs = next_obs

		episode_step += 1

	print('Completed training for {}. Total time {:.3f}'.format(work_dir, time.time()-START))

if __name__ == "__main__":
    args = parse_args()
    main(args)
