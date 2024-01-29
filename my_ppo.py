import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    
class Memory:
    def __init__(self):
        self.actions = []
        self.state_1 = []
        self.state_2 = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.state_1[:]
        del self.state_2[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
    
class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.features1 = nn.Sequential(
            nn.Linear(state_dim, 64), # state dim 6
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Tanh()
        )
        
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, stride=2, padding=0), # (图像尺寸-卷积核尺寸 + 2*填充值)/步长+1
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(9*9*40, 16),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(48, 2),  # action ax,  ay
            nn.Tanh(),
        )
    def forward(self, s1, s2):
        x1 = self.features1(s1)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.features2(s2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc1(x2)
        x = torch.cat((x1, x2), dim=1)
        a = self.fc2(x)
        return a

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.features1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh()
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=4, stride=2, padding=0), # (图像尺寸-卷积核尺寸 + 2*填充值)/步长+1
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(9*9*40, 16),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(48, 1)
        )

    def forward(self, s1, s2):
        x1 = self.features1(s1)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.features2(s2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc1(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc2(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_var):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.action_var = torch.tensor(action_var).to(device)
        # actor
        self.actor = Actor(state_dim)
        # critic
        self.critic = Critic(state_dim)


    def act(self, state_1, state_2, memory): #
        action_mean = self.actor(state_1[:,2:,], state_2)
        cov_mat = torch.diag(self.action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        memory.state_1.append(state_1)
        memory.state_2.append(state_2)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.detach()
  
    def evaluate(self, state_1, state_2, action): #
        action_mean = self.actor(state_1[:,2:], state_2)
        action_var = self.action_var.expand_as(action_mean)
        # torch.diag_embed(input, offset=0, dim1=-2, dim2=-1) → Tensor
        # Creates a tensor whose diagonals of certain 2D planes (specified by dim1 and dim2) are filled by input
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state_1[:,2:,], state_2)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        # self.v_size = vis_pix * 2
        self.v_size = 20 * 2
        self.policy = ActorCritic(state_dim, action_dim, action_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, action_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state_1, state_2, memory):
        state_1 = torch.FloatTensor(state_1.reshape(1, -1)).to(device)
        state_2 = torch.FloatTensor(state_2.reshape((1, 1, self.v_size, self.v_size))).to(device)
        return self.policy_old.act(state_1, state_2, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_state_1 = torch.squeeze(torch.stack(memory.state_1).to(device), 1).detach()
        old_state_2 = torch.squeeze(torch.stack(memory.state_2).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_state_1, old_state_2, old_actions)
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())