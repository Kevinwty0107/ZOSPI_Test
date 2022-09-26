import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import mdn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
torch.cuda.set_device(3)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


    
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class SPI(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        phi = 0.0,
        NoG = 20
    ):
        self.NoG = NoG
        
        if self.NoG >=2:
            print('using Mixture of Gaussian policy, number of Gaussian:',self.NoG)
            self.actor =nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            mdn.MDN(64, action_dim, self.NoG)
                        ).to(device)
            self.actor.max_action = max_action
        else:
            print('using Gaussian Policy')
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state): # this function is only called when evaluation, so it's better to be deterministic, i.e., the mean
        if self.NoG >=2:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            mean, var, w = self.actor(state)
            var *= 0.0
            y_test = mdn.sample(mean, var, w)
            return y_test.cpu().data.numpy().flatten()
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            return self.actor(state).cpu().data.numpy().flatten()

    def sample_action_and_select_max(self, state, action, mode = 'random', batch_size=64, sample_rep_num = 50, bcq_net = None, expl_noise = 0.5):
        #embed()
        act_array = np.zeros((batch_size, sample_rep_num, action[0].shape[0]))
        value = np.zeros((batch_size, sample_rep_num))
        max_idx = np.zeros((batch_size,)).astype(int)
        if self.NoG >= 2:
            mean, var, w = self.actor_target(state)
            var *= 0.0
            act_tgt = mdn.sample(mean, var, w)
            value_baseline = self.critic_target.Q1(state, act_tgt).detach().cpu().numpy()
        else:    
            value_baseline = self.critic_target.Q1(state, self.actor_target(state)).detach().cpu().numpy()
        mask = np.zeros((batch_size,))

        act_max_array = np.zeros((batch_size, action[0].shape[0]))
        if mode == 'fixed':
            sample_rep_num = self.act_array_fixed.shape[0]
            act_array = self.act_array_fixed
            act_array = act_array.reshape([1,act_array.shape[0],act_array.shape[1]]).repeat(batch_size,0)

            value = self.critic_target.Q1(state.unsqueeze(1).expand([-1,sample_rep_num,-1]).reshape(-1,state[0].shape[0]).float().to(device),torch.as_tensor(act_array).reshape(-1,action[0].shape[0]).float().to(device)).detach().cpu().numpy().reshape(batch_size,sample_rep_num,-1)
            max_idx = np.argmax(value,axis=-2)
            for i in range(batch_size):
                act_max_array[i] = act_array[i,int(max_idx[i]),:]
                mask[i] = 1.0 if value_baseline[i] < value[i][int(max_idx[i])] else 0.0
            #print('time used',dt.now()-start_time)
            return act_max_array, mask



        if mode == 'random':
            act_array = ((np.random.random(act_array.shape)-0.5)*action[0].shape[0])*self.actor.max_action
            #embed()

            value = self.critic_target.Q1(state.unsqueeze(1).expand([-1,sample_rep_num,-1]).reshape(-1,state[0].shape[0]).float().to(device),torch.as_tensor(act_array).reshape(-1,action[0].shape[0]).float().to(device)).detach().cpu().numpy().reshape(batch_size,sample_rep_num,-1)
            max_idx = np.argmax(value,axis=-2)
            for i in range(batch_size):
                act_max_array[i] = act_array[i,int(max_idx[i]),:]
                mask[i] = 1.0 if value_baseline[i] < value[i][int(max_idx[i])] else 0.0
            #print('time used',dt.now()-start_time)
            return act_max_array, mask

        if mode == 'onpolicy':
            if self.NoG >=2:
                mean, var, w = self.actor_target(state)
                var *= 0.0
                action = mdn.sample(mean, var, w).detach().cpu().numpy()
            else:
                action = self.actor_target(state).detach().cpu().numpy()
            #embed()

            act_array = (((np.random.random(act_array.shape)-0.5)*expl_noise)*self.actor.max_action + action.reshape(batch_size,1,action[0].shape[0]).repeat(sample_rep_num,1)).clip(-self.actor.max_action, self.actor.max_action)

            value = self.critic_target.Q1(state.unsqueeze(1).expand([-1,sample_rep_num,-1]).reshape(-1,state[0].shape[0]).float().to(device),torch.as_tensor(act_array).reshape(-1,action[0].shape[0]).float().to(device)).detach().cpu().numpy().reshape(batch_size,sample_rep_num,-1)
            max_idx = np.argmax(value,axis=-2)
            for i in range(batch_size):
                act_max_array[i] = act_array[i,int(max_idx[i]),:]
                mask[i] = 1.0 if value_baseline[i] < value[i][int(max_idx[i])] else 0.0
            return act_max_array, mask
        if mode == 'random_local':
            act_array = ((np.random.random(act_array.shape)-0.5)*action[0].shape[0])*self.actor.max_action
            #embed()
            
            if self.NoG >=2:
                mean, var, w = self.actor_target(state)
                var *= 0.0
                action = mdn.sample(mean, var, w).detach().cpu().numpy()
            else:
                action = self.actor_target(state).detach().cpu().numpy()
            
            
            
            act_array_2 = (((np.random.random(act_array.shape)-0.5)*expl_noise)*self.actor.max_action + action.reshape(batch_size,1,action[0].shape[0]).repeat(sample_rep_num,1)).clip(-self.actor.max_action, self.actor.max_action)
            act_array = np.hstack((act_array,act_array_2))

            value = self.critic_target.Q1(state.unsqueeze(1).expand([-1,sample_rep_num*2,-1]).reshape(-1,state[0].shape[0]).float().to(device),torch.as_tensor(act_array).reshape(-1,action[0].shape[0]).float().to(device)).detach().cpu().numpy().reshape(batch_size,sample_rep_num*2,-1)
            max_idx = np.argmax(value,axis=-2)
            for i in range(batch_size):
                act_max_array[i] = act_array[i,int(max_idx[i]),:]
                mask[i] = 1.0 if value_baseline[i] < value[i][int(max_idx[i])] else 0.0
            #print('time used',dt.now()-start_time)
            return act_max_array, mask


    def train_supervised_with_phi(self, replay_buffer, batch_size=100, sample_rep = 50, actor_training_time = 1, sample_mode = 'random',bcq=None):

        for _ in range(10):
            self.total_it += 1
            self.counter_mask = []
            # Sample replay buffer
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                if self.NoG>=2:
                    mean, var, w = self.actor_target(next_state)
                    var *= 0.0
                    next_action = (
                        mdn.sample(mean, var, w) + noise
                    ).clamp(-self.max_action, self.max_action)
                else:
                    next_action = (
                        self.actor_target(next_state) + noise
                    ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q


            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


            if self.total_it % self.policy_freq == 0:
                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                


            state_val, action_val, next_state_val, reward_val, not_done_val = replay_buffer.sample(batch_size)


            for i in range(actor_training_time):
                '''
                different from DDPG train method, we use supervised learning here to update the actor
                '''

                action_alt, mask = self.sample_action_and_select_max(state,action,sample_mode, batch_size, sample_rep,bcq_net = bcq)
                action_alt_val, mask_val = self.sample_action_and_select_max(state_val,action_val,sample_mode, batch_size, sample_rep,bcq_net = bcq)
                if self.NoG>=2:
                    #for minibatch, labels in train_set:
                    self.actor_optimizer.zero_grad()
                    pi, sigma, mu = self.actor(state)
                    loss = mdn.mdn_loss(pi, sigma, mu, torch.as_tensor(action_alt).float().to(device))
                    loss.backward()
                    self.actor_optimizer.step()
                else:
                    actor_loss = torch.mean(torch.mean((self.actor(state) - torch.as_tensor(action_alt).float().to(device))**2, 1) * torch.as_tensor(mask).float().to(device))

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()


                actor_loss_val = None #torch.mean(torch.mean((self.actor(state_val) - torch.as_tensor(action_alt_val).float().to(device))**2, 1) * torch.as_tensor(mask_val).float().to(device))

                self.counter_mask.append(np.sum(mask)/256)
                if self.total_it % self.policy_freq == 0:
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return self.counter_mask, actor_loss_val


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
