import os
import json
import copy
import dill
import random
import torch
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from rl.env import Env
from rl.dataset import PPODataset


class Agent():
    def __init__(self, model, config):
        self.config = config

        if 'test' != self.config.exp_name:
            self.config.log_dir = os.path.join('rl/log', self.config.dataset, datetime.now().strftime('%b%d_%H_%M') + self.config.exp_name)
        else:
            self.config.log_dir = os.path.join('rl/log', self.config.exp_name)
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)
        with open(os.path.join(self.config.log_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

        print(self.config.log_dir, self.config.device, self.config.dataset)

        np.random.seed(self.config.seed)#固定随机数种子
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
        self.model = model.to(self.config.device)
        self.model.eval()#不使用BN和dropout
        self.ref_model = copy.deepcopy(model).to(self.config.device)
        self.ref_model.eval()#不使用BN和dropout

        with open('rl/data/{:}-{:}.pkl'.format(self.config.dataset, 'train'), 'rb') as f:
            self.train_dataset = dill.load(f, encoding='latin1')
        with open('rl/data/{:}-{:}.pkl'.format(self.config.dataset, 'eval'), 'rb') as f:
            self.eval_dataset = dill.load(f, encoding='latin1')

        self.env = Env(self.train_dataset, model, self.config)
        if self.config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.diffusion.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'SGD':
            self.optimizer = torch.optim.Adam(self.model.diffusion.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)

        self.contexts   = torch.zeros(100, self.config.worker * self.config.sample, 256).to(self.config.device)
        self.xts        = torch.zeros(100, self.config.worker * self.config.sample, 12, 2).to(self.config.device)
        self.betas      = torch.zeros(100, self.config.worker * self.config.sample)

        self.mus        = torch.zeros(100, self.config.worker * self.config.sample, 12, 2).to(self.config.device)#net 的output
        self.sigmas     = torch.zeros(100, self.config.worker * self.config.sample, 12, 2)
        self.actions    = torch.zeros(100, self.config.worker * self.config.sample, 12, 2).to(self.config.device)#e_theta + sigma/c0/c1 * z
        self.ref_rewards= torch.zeros(100, self.config.worker * self.config.sample).to(self.config.device)
        self.rewards    = torch.zeros(100, self.config.worker * self.config.sample).to(self.config.device)
        self.advantages = torch.zeros(100, self.config.worker * self.config.sample).to(self.config.device)

        self.betas = self._betas().to(self.config.device)
        self.sigmas = self._sigma().to(self.config.device)
        self.dataloader = None


    def _betas(self):
        betas = self.model.diffusion.var_sched.betas
        betas = betas[range(100, 0, -1)]
        betas = betas.unsqueeze(1).repeat(1, self.config.worker * self.config.sample)
        return betas


    def _sigma(self):
        alphas = self.model.diffusion.var_sched.alphas
        alpha_bars = self.model.diffusion.var_sched.alpha_bars
        sigmas = self.model.diffusion.var_sched.sigmas_inflex

        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1 - alphas) / torch.sqrt(1 - alpha_bars)
        s = sigmas/c0/c1

        s = s[range(100, 0, -1)]
        s[-1] = 0.01#方差不能为0设置为一个很小的数字
        s = s.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, self.config.worker * self.config.sample, 12, 2)
        return s


    def train_actions(self):
        with torch.no_grad():
            context, xt = self.env.reset()
            context = context.to(self.config.device)
            xt = xt.to(self.config.device)

            self.contexts = context.unsqueeze(0).repeat(100, 1, 1).to(self.config.device)

            for step in range(100):
                mu = self.model.diffusion.net(xt, beta=self.betas[step], context=context)
                ref_mu = self.ref_model.diffusion.net(xt, beta=self.betas[step], context=context)
                self.xts[step] = xt
                self.mus[step] = mu
                self.actions[step] = action = mu + self.sigmas[step] * torch.randn_like(mu)
                self.ref_rewards[step] = (((action - mu) ** 2 - (action - ref_mu) ** 2) / (2 * self.sigmas[step] ** 2)).sum(-1).sum(-1)

                xt, reward = self.env.step(action)
                xt = xt.to(self.config.device)
                self.rewards[step] = reward
            
            ade, fde = self.env.caculate_ade_fde()
            with open(os.path.join(self.config.log_dir, 'log'), 'a') as f:
                f.write('ade, fde:{:.4f}\t{:.4f}\n'.format(ade, fde))


    def caculate_buffer(self):
        with torch.no_grad():
            for t in reversed(range(100)):
                if t == 99:
                    self.advantages[t] = self.rewards[t]
                else:
                    self.advantages[t] = self.rewards[t] + self.config.gamma * self.advantages[t + 1]

            self.ref_rewards = (self.ref_rewards - self.ref_rewards.mean()) / (self.ref_rewards.std(unbiased=False) + 1e-8)
            self.advantages += self.config.ceof_ref * self.ref_rewards
            
            batchsize = 100 * self.config.worker * self.config.sample // self.config.num_minibatches
            self.dataloader = DataLoader(PPODataset(self), batch_size=batchsize, shuffle=True)
    

    def update(self):
        for epoch in range(self.config.update_epoch):
            clip_true = 0
            clip_false = 0
            loss_true = 0
            loss_false = 0
            ratio_abs_true = 0
            ratio_abs_false = 0
            right = 0

            for contexts, xts, betas, old_mus, sigmas, actions, advantages in self.dataloader:
                new_mus = self.model.diffusion.net(xts, beta=betas, context=contexts)

                log_pi_old = Normal(old_mus, sigmas).log_prob(actions).sum(-1).sum(-1)#TODO:dataset处理有问题， context和old_mus不对齐
                log_pi_new = Normal(new_mus, sigmas).log_prob(actions).sum(-1).sum(-1)
                ratio = (log_pi_new - log_pi_old).exp()
                clip = (advantages > 0) * (ratio < 1 + self.config.clip_coef) + (advantages < 0) * (ratio > 1 - self.config.clip_coef)

                loss = - (clip * advantages * ratio).sum()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gard_norm)
                self.optimizer.step()

                ###log
                ratio = ratio.data
                loss = - advantages * ratio

                clip_true += (clip == True).sum()
                clip_false += (clip == False).sum()
                loss_true += loss[clip == True].sum()
                loss_false += loss[clip == False].sum()
                ratio_abs_true += (ratio[clip == True] - 1).abs().sum()
                ratio_abs_false += (ratio[clip == False] - 1).abs().sum()
                right += ((advantages > 0) * (ratio - 1) + (advantages < 0) * (1 - ratio)).sum()
            
            clip = clip_true / (clip_true + clip_false)
            loss = (loss_true + loss_false) / (clip_true + clip_false)
            loss_true /= clip_true
            loss_false /= clip_false
            ratio = (ratio_abs_true + ratio_abs_false) / (clip_true + clip_false)
            ratio_abs_true /= clip_true
            ratio_abs_false /= clip_false
            right /= (clip_true + clip_false)
            
            
            with open(os.path.join(self.config.log_dir, 'log'), 'a') as f:
                f.write('{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(clip, loss, loss_true, loss_false, ratio, ratio_abs_true, ratio_abs_false, right))
                if epoch == self.config.update_epoch - 1:
                    f.write('\n')
        
        self.scheduler.step()

    def test(self, mode='eval'):
        if mode == 'train':
            dataloader = DataLoader(self.train_dataset, batch_size=1024)
        elif mode == 'eval':
            dataloader = DataLoader(self.eval_dataset, batch_size=1024)



        ades, fdes = [], []
        for context, y, _, _ in tqdm(dataloader):
            context = context.to(self.config.device)
            y = y.to(self.config.device)

            predicted_y_vel =  self.diffusion(context)
            predicted_y_pos_rel = torch.cumsum(predicted_y_vel, dim=2) * 0.4
            predicted_y_pos_rel = predicted_y_pos_rel.detach()

            ade = (((predicted_y_pos_rel - y) ** 2).sum(-1)**0.5).mean(-1).min(0)[0]
            fde = (((predicted_y_pos_rel[:, :, -1] - y[:,  -1])**2).sum(-1)**0.5).min(0)[0]
            ades.append(ade)
            fdes.append(fde)
        
        ade = torch.cat(ades)
        fde = torch.cat(fdes)
        ade = ade.mean()
        fde = fde.mean()

        if self.config.dataset == "eth":
            ade, fde =  ade / 0.6, fde / 0.6
        elif self.config.dataset == "sdd":
            ade, fde =  ade * 50, fde * 50

        with open(os.path.join(self.config.log_dir, 'log'), 'a') as f:
            f.write('test:{:.4f}\t{:.4f}\n'.format(ade, fde))


    def diffusion(self, context):
        alphas = self.model.diffusion.var_sched.alphas
        alpha_bars = self.model.diffusion.var_sched.alpha_bars
        betas = self.model.diffusion.var_sched.betas
        sigmas = self.model.diffusion.var_sched.sigmas_inflex

        traj_list = []
        for i in range(20):
            batch_size = context.size(0)
            x_T = torch.randn([batch_size, 12, 2])
            traj = torch.zeros(101, batch_size, 12, 2).to(self.config.device)
            traj[100] = x_T
            for t in range(100, 0, -1):
                x_t = traj[t]
                beta = betas[[t] * batch_size]
                e_theta = self.model.diffusion.net(x_t, beta=beta, context=context)
                z = torch.randn_like(x_T).to(self.config.device) if t > 1 else torch.zeros_like(x_T).to(self.config.device)

                c0 = 1.0 / torch.sqrt(alphas[t])
                c1 = (1 - alphas[t]) / torch.sqrt(1 - alpha_bars[t])
                sigma = sigmas[t]

                # if t <= 5:
                #     sigma *= 0.001
                
                x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                traj[t-1] = x_next.data
       
            traj_list.append(traj[0])
        return torch.stack(traj_list)
