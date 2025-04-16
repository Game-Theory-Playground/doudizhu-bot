# rarsms_irl_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
from trainers import BaseTrainer
from bots import RARSMSBot, DMCBot
from irl.reward_net import Discriminator
from irl.expert_dataset import ExpertDataset
from rlcard.utils import set_seed

class RARSMSIRLTrainer(BaseTrainer):
    def __init__(self, env, savedir, expert_data_path,
                 cuda='', save_interval=10, 
                 training_device='0',
                 gamma=0.99, lr=1e-4, use_dmc=False, dmc_model_path=None):
        super().__init__(env, savedir)
        self.device = torch.device(f'cuda:{training_device}' if torch.cuda.is_available() and cuda != '' else 'cpu')
        self.gamma = gamma
        self.lr = lr
        self.save_interval = save_interval

        if use_dmc:
            self.agent = DMCBot(dmc_model_path, position=0, device=self.device)
        else:
            self.agent = RARSMSBot(None, position=0)
            self.agent.set_device(self.device)

        self.discriminator = Discriminator().to(self.device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.optimizer_policy = optim.Adam(self.agent.actor.parameters(), lr=self.lr) if hasattr(self.agent, 'actor') else None

        self.expert_dataset = ExpertDataset(expert_data_path)

    def _train_discriminator(self, expert_batch, policy_batch):
        expert_s, expert_a = zip(*expert_batch)
        policy_s, policy_a = zip(*policy_batch)

        s = torch.cat([torch.stack(expert_s), torch.stack(policy_s)], dim=0).to(self.device)
        a = torch.cat([torch.tensor(expert_a), torch.tensor(policy_a)], dim=0).to(self.device)
        labels = torch.cat([
            torch.ones(len(expert_batch)),
            torch.zeros(len(policy_batch))
        ]).to(self.device)

        logits = self.discriminator(s, a).squeeze()
        loss = nn.BCELoss()(logits, labels)
        self.optimizer_d.zero_grad()
        loss.backward()
        self.optimizer_d.step()

        return loss.item()

    def _train_policy(self, trajectories):
        if not hasattr(self.agent, 'actor'):
            return 0.0, 0.0

        actor_loss = 0
        critic_loss = 0

        for (s, a, _, next_s, done) in trajectories:
            s = s.to(self.device)
            a = torch.tensor([a]).to(self.device)
            r = self.discriminator.reward(s, a).detach()

            value = self.agent.critic(s)
            with torch.no_grad():
                next_value = self.agent.critic(next_s.to(self.device))
                target = r + self.gamma * next_value * (1 - int(done))

            loss_critic = nn.MSELoss()(value, target)

            log_prob = torch.log(self.agent.actor.get_prob(s)[a])
            advantage = (target - value).detach()
            loss_actor = -log_prob * advantage

            self.optimizer_policy.zero_grad()
            (loss_actor + loss_critic).backward()
            self.optimizer_policy.step()

            actor_loss += loss_actor.item()
            critic_loss += loss_critic.item()

        return actor_loss / len(trajectories), critic_loss / len(trajectories)

    def train(self):
        set_seed(42)
        episode = 0
        all_trajectories = []

        while True:
            traj = []
            s = self.env.reset()[0]['obs']
            done = False

            while not done:
                a = self.agent.act(s)
                next_state, reward, done, _ = self.env.step(a)
                s_next = next_state[0]['obs']
                traj.append((torch.tensor(s), a, reward, torch.tensor(s_next), done))
                s = s_next

            all_trajectories += traj

            expert_batch = self.expert_dataset.sample(batch_size=32)
            policy_batch = random.sample([(s, a) for s, a, _, _, _ in traj], 32)

            loss_d = self._train_discriminator(expert_batch, policy_batch)
            loss_a, loss_c = self._train_policy(traj)

            if episode % self.save_interval == 0 and hasattr(self.agent, 'actor'):
                torch.save(self.agent.actor.state_dict(), f'{self.savedir}/actor_{episode}.pth')
                torch.save(self.agent.critic.state_dict(), f'{self.savedir}/critic_{episode}.pth')

            print(f"[Episode {episode}] D_loss={loss_d:.4f} A_loss={loss_a:.4f} C_loss={loss_c:.4f}")
            episode += 1
