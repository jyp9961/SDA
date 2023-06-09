import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def read_eval(dirs, min_episode=500, eval_freq=10):
    # the experiment should run at least 'min_episode' episodes
    all_episodes = []
    all_normal_rewards = []
    all_color_rewards = []
    all_video_easy_rewards = []
    all_video_rewards = []
    for filename in dirs:
        eval_filename = os.path.join(filename,'eval.csv')
        if os.path.exists(eval_filename):
            df = pd.read_csv(eval_filename)

            if df['episode'].values[-1] >= min_episode:
                all_episodes.append(df['episode'].values)
                all_normal_rewards.append(df['episode_reward_normal'].values)
                all_color_rewards.append(df['episode_reward_color_hard'].values)
                all_video_easy_rewards.append(df['episode_reward_video_easy'].values)
                all_video_rewards.append(df['episode_reward_video_hard'].values)
        else:
            print('{} doesn\'t exist.'.format(eval_filename))

    #print('episodes', [int(episode[-1]) for episode in all_episodes])
    return all_episodes, all_normal_rewards, all_color_rewards, all_video_easy_rewards, all_video_rewards

def compute_mean_std(data):
    data_length = [len(d) for d in data]
    mean = []
    std = []
    for i in range(max(data_length)):
        tmp = []
        for j in range(len(data)):
            if i < data_length[j]:
                tmp.append(data[j][i])
        mean.append(np.mean(tmp))
        std.append(np.std(tmp))
    
    return np.array(mean), np.array(std)

def plot_rewards(dirs, exp_name, color=None, eval_freq=10, eval_num=1, plot_std=True):
    all_episodes, all_normal_rewards, all_color_rewards, all_video_easy_rewards, all_video_rewards = read_eval(dirs, min_episode=50, eval_freq=eval_freq)
    plot_episodes = np.arange(max([len(_) for _ in all_episodes])) * eval_freq
    
    last_normal_rewards = [x for _ in all_normal_rewards for x in _[-eval_num:]]
    last_color_rewards = [x for _ in all_color_rewards for x in _[-eval_num:]]
    last_video_easy_rewards = [x for _ in all_video_easy_rewards for x in _[-eval_num:]]
    last_video_rewards = [x for _ in all_video_rewards for x in _[-eval_num:]]

    # normal eval rewards
    plt.subplot(241)
    plot_normal_rewards_mean, plot_normal_rewards_std = compute_mean_std(all_normal_rewards)
    plt.plot((plot_episodes+all_episodes[0][0])*1000, plot_normal_rewards_mean, label=exp_name)
    if plot_std:
        plt.fill_between((plot_episodes+all_episodes[0][0])*1000, plot_normal_rewards_mean-plot_normal_rewards_std, plot_normal_rewards_mean+plot_normal_rewards_std, alpha=0.4)
    print('normal bg, {}: {}; mean {:.2f} +- std {:.2f}'.format(exp_name, [len(_) for _ in all_normal_rewards], np.mean(last_normal_rewards), np.std(last_normal_rewards)))
    
    # color_hard eval rewards
    plt.subplot(242)
    plot_color_rewards_mean, plot_color_rewards_std = compute_mean_std(all_color_rewards)
    plt.plot((plot_episodes+all_episodes[0][0])*1000, plot_color_rewards_mean, label=exp_name)
    if plot_std:
        plt.fill_between((plot_episodes+all_episodes[0][0])*1000, plot_color_rewards_mean-plot_color_rewards_std, plot_color_rewards_mean+plot_color_rewards_std, alpha=0.4)
    print('color-hard bg, {}: {}; mean {:.2f} +- std {:.2f}'.format(exp_name, [len(_) for _ in all_normal_rewards], np.mean(last_color_rewards), np.std(last_color_rewards)))

    # video_easy eval_rewards
    plt.subplot(243)
    plot_video_easy_rewards_mean, plot_video_easy_rewards_std = compute_mean_std(all_video_easy_rewards)
    plt.plot((plot_episodes+all_episodes[0][0])*1000, plot_video_easy_rewards_mean, label=exp_name)
    if plot_std:
        plt.fill_between((plot_episodes+all_episodes[0][0])*1000, plot_video_easy_rewards_mean-plot_video_easy_rewards_std, plot_video_easy_rewards_mean+plot_video_easy_rewards_std, alpha=0.4)
    print('video-easy bg, {}: {}; mean {:.2f} +- std {:.2f}'.format(exp_name, [len(_) for _ in all_normal_rewards], np.mean(last_video_easy_rewards), np.std(last_video_easy_rewards)))

    # video_hard eval_rewards
    plt.subplot(244)
    plot_video_rewards_mean, plot_video_rewards_std = compute_mean_std(all_video_rewards)
    plt.plot((plot_episodes+all_episodes[0][0])*1000, plot_video_rewards_mean, label=exp_name)
    if plot_std:
        plt.fill_between((plot_episodes+all_episodes[0][0])*1000, plot_video_rewards_mean-plot_video_rewards_std, plot_video_rewards_mean+plot_video_rewards_std, alpha=0.4)
    print('video-hard bg, {}: {}; mean {:.2f} +- std {:.2f}'.format(exp_name, [len(_) for _ in all_normal_rewards], np.mean(last_video_rewards), np.std(last_video_rewards)))

    # ratio, color_hard / normal 
    plt.subplot(246)
    color_hard_ratio = plot_color_rewards_mean / plot_normal_rewards_mean
    plt.plot((plot_episodes+all_episodes[0][0])*1000, color_hard_ratio, label=exp_name)
    
    # ratio, video_easy / normal 
    plt.subplot(247)
    video_easy_ratio = plot_video_easy_rewards_mean / plot_normal_rewards_mean
    plt.plot((plot_episodes+all_episodes[0][0])*1000, video_easy_ratio, label=exp_name)
    
    # ratio, video_hard / normal 
    plt.subplot(248)
    video_hard_ratio = plot_video_rewards_mean / plot_normal_rewards_mean
    plt.plot((plot_episodes+all_episodes[0][0])*1000, video_hard_ratio, label=exp_name)
    
    return np.mean(last_normal_rewards), np.std(last_normal_rewards), np.mean(last_color_rewards), np.std(last_color_rewards), np.mean(last_video_easy_rewards), np.std(last_video_easy_rewards), np.mean(last_video_rewards), np.std(last_video_rewards)
    
def plt_dir_rewards(dir_name, exp_name, color=None, eval_freq=10, eval_num=1, plot_std=True):
    # eval_freq: evaluate every n episodes

    dirs = os.listdir(dir_name)
    for i in range(len(dirs)):
        dirs[i] = os.path.join(dir_name, dirs[i])
    normal_mean, normal_std, color_mean, color_std, video_easy_mean, video_easy_std, video_mean, video_std = plot_rewards(dirs, exp_name, color, eval_freq, eval_num, plot_std)

    return normal_mean, normal_std, color_mean, color_std, video_easy_mean, video_easy_std, video_mean, video_std

if __name__ == '__main__':
    env_names = [['walker', 'walk'], ['walker', 'stand'], ['cartpole', 'swingup'], ['ball_in_cup', 'catch'], ['finger','spin']]
    plot_type = 'AU7005'
    
    if plot_type == 'AU7005':
        algo_names = ['svea', 'sda_quantile0.95', 'sda_quantile0.9']
        label_dicts = {}
        for algo_name in algo_names: label_dicts[algo_name] = algo_name
        latex_name = 'table_result_latex'
        csv_name = 'dmgb_table.csv'
    
    cols = ['env_name', 'svea(normal)', 'sgsac(normal)', 'svea(color-hard)', 'sgsac(color-hard)', 'svea(video-easy)', 'sgsac(video-easy)', 'svea(video-hard)', 'sgsac(video-hard)']
    table_result_df = pd.DataFrame(columns = cols)
    
    work_dir = 'results'    
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    work_dir = os.path.join(work_dir, plot_type)
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    
    eval_num = 10
    plot_std = False
    for env_name in env_names:
        print(env_name)
        textbfs = ''
        plt.figure(figsize=(20,10))
        domain_name, task_name = env_name
        result_dir = '{}_{}_results_onlycsv'.format(domain_name, task_name)
        for algo_name in algo_names:
            algo_dir = os.path.join(result_dir, algo_name)
            if os.path.exists(algo_dir):
                label_name = label_dicts[algo_name]
                try:
                    normal_mean, normal_std, color_mean, color_std, video_easy_mean, video_easy_std, video_mean, video_std = plt_dir_rewards(algo_dir, label_name, eval_num=eval_num, plot_std=plot_std)
                except Exception as e:
                    normal_mean, normal_std, color_mean, color_std, video_easy_mean, video_easy_std, video_mean, video_std = 0, 0, 0, 0, 0, 0, 0, 0
                    print(e)
        
        exp_dict = {1:'normal',2:'color_hard',3:'video_easy',4:'video_hard',6:'color-hard/normal', 7:'video-easy/normal', 8:'video-hard/normal'}
        for i in exp_dict:
            plt.subplot(240+i)
            plt.xlabel('environment steps')
            if i <= 4:
                plt.ylabel('episode_reward (eval)')
            elif 6 <= i <= 8:
                plt.ylabel('ratio')
            #plt.ylim(0, 1000)
            plt.legend(loc='best')
            plt.grid()
            plt.title('{}_{} {} eval'.format(domain_name, task_name, exp_dict[i]))
            plt.savefig(os.path.join(work_dir,'{}_{}_gb'.format(domain_name, task_name)))
        plt.close()
    
    print(table_result_df)