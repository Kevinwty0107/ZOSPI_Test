import matplotlib.pyplot as plt
import numpy as np
import torch
import gym
from argparse import ArgumentParser
import os
import torch.nn.functional as F
import utils
import SPI

def parse_args():
    parser = ArgumentParser(description='train args')
    #parser.add_argument('config', help='config file path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-en','--env_name', type=str, default=None)
    parser.add_argument('-a','--alias', type=str, default=None)
    parser.add_argument('-sr','--sample_reuse', type=int, default=10)
    parser.add_argument('-b','--batch_size', type=int, default=25)
    parser.add_argument('-ts','--max_timestep', type = int, default = 1000000)
    parser.add_argument('-r','--repeat',type=int,default=None)
    parser.add_argument('-eps','--epsilon',type=float,default=0.0)
    parser.add_argument('-ns','--number_sample',type=int, default=50)
    parser.add_argument('-phi','--phi',type=float,default=0.05)
    parser.add_argument('-sm','--sample_mode',type=str,default='random_local')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    ENV_NAME = args.env_name
    alias = f'BS_{args.batch_size}_NS_{args.number_sample}_EPS_{args.epsilon}_EXTAlias_{args.alias}_REPEAT_{args.repeat}'

    def eval_policy(policy, eval_episodes=10):
        eval_env = gym.make(ENV_NAME)

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = policy.select_action(np.array(state))
                state, reward, done,_ = eval_env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes
        #print("---------------------------------------")
        #print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        #print("---------------------------------------")
        return avg_reward

    env = gym.make(ENV_NAME)
    os.makedirs('results',exist_ok=True)
    os.makedirs('ckpts',exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    args_policy_noise = 0.2
    args_noise_clip = 0.5
    args_policy_freq = 2
    args_max_timesteps = args.max_timestep
    args_expl_noise = 0.1
    args_batch_size = args.batch_size
    args_eval_freq = 1000
    args_start_timesteps = 25000

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": 0.99,
        "tau": 0.005,
        "phi":args.phi
    }

    for repeat in range(args.repeat, args.repeat+1):

        if True:
            args_policy = 'TD3'

            if args_policy == "TD3":
                # Target policy smoothing is scaled wrt the action scale
                kwargs["policy_noise"] = args_policy_noise * max_action
                kwargs["noise_clip"] = args_noise_clip * max_action
                kwargs["policy_freq"] = args_policy_freq
                policy = SPI_Exploitation_WithLocal.TD3(**kwargs)
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

        # Evaluate untrained policy
        evaluations = [eval_policy(policy)]
        indi_list = []
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        counter = 0
        msk_list = []        
        temp_curve = [eval_policy(policy)]
        temp_val = []
        for t in range(int(args_max_timesteps)):
            episode_timesteps += 1
            counter += 1
            # Select action randomly or according to policy
            if t < args_start_timesteps:
                action = np.random.uniform(-max_action,max_action,action_dim)
            else:
                if np.random.uniform(0,1) < args.epsilon:
                    action = np.random.uniform(-max_action,max_action,action_dim)
                else:
                    action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args_expl_noise, size=action_dim)
                    ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done,_ = env.step(action) 
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            if t >= args_start_timesteps:
                '''TD3'''
                last_val = 999.
                patient = 5
                sub_indi = []
                for i in range(args.sample_reuse):
                    ___,___, lg_indi = policy.train_supervised_with_phi(replay_buffer, args_batch_size, sample_mode =args.sample_mode, sample_rep = args.number_sample)
                    sub_indi.append(lg_indi)
                    '''Supervised Learning'''
                indi_list.append(sub_indi)
                        
            # Train agent after collecting sufficient data
            if done: 
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                msk_list = []
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % args_eval_freq == 0:
                evaluations.append(eval_policy(policy))
                print('recent Evaluation:',evaluations[-1])
                if (t + 1) % (args_eval_freq * 10) == 0:
                    policy.save("ckpts/ckpt_{}_env_{}_repeat_{}_step{}".format(alias,ENV_NAME,repeat,t + 1))
                np.save('results/evaluations_alias{}_ENV{}_Repeat{}'.format(alias,ENV_NAME,repeat),evaluations)
                np.save('results/selected_indi_alias{}_ENV{}_Repeat{}'.format(alias,ENV_NAME,repeat),indi_list)
            