import os
import torch

class Env():
    def __init__(self, dataset, model, config):
        self.dataset = dataset
        self.model = model
        self.config = config

        self.c0, self.c1 = self._c()
        self.now_ade = self.dataset.ade.min(-1)[0].to(self.config.device)
        self.index = None
        self.context, self.y, self.ref_ade, self.ref_fde = None, None, None, None
        self.xt = None
        self.t = None
        self.done = False

    def _c(self):
        alphas = self.model.diffusion.var_sched.alphas
        alpha_bars = self.model.diffusion.var_sched.alpha_bars
        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1 - alphas) / torch.sqrt(1 - alpha_bars)
        return c0, c1

    def reset(self):
        if self.config.reset == 'overfit':
            self.index = torch.arange(self.config.worker)
        elif self.config.reset == 'random':
            self.index = torch.randint(0, len(self.dataset), (self.config.worker, ))
        elif self.config.reset == 'ade_rank':
            probs = 1 / (self.now_ade.argsort(descending=True).argsort() + 1)
            probs = probs / probs.sum()
            self.index = torch.multinomial(probs, self.config.worker, replacement=True)
        elif self.config.reset == 'ade':
            probs = self.now_ade
            probs = probs / probs.sum()
            self.index = torch.multinomial(probs, self.config.worker, replacement=True)
        elif self.config.reset == 'half_ade':
            index1 = self.index = torch.randint(0, len(self.dataset), (self.config.worker // 2, )).to(self.config.device)

            probs = self.now_ade
            probs = probs / probs.sum()
            index2 = torch.multinomial(probs, self.config.worker // 2, replacement=True)

            self.index = torch.cat([index1,index2])
        
        self.context, self.y, self.ref_ade, self.ref_fde = self.dataset[self.index]

        self.context = self.context.unsqueeze(1).repeat(1, self.config.sample, 1)#(self.config.worker , self.config.sample , 256)
        self.y = self.y.unsqueeze(1).repeat(1, self.config.sample, 1, 1)#(self.config.worker , self.config.sample , 12, 2)
        self.context = self.context.reshape(self.config.worker * self.config.sample, 256).to(self.config.device)#[a,a,a,a,b,b,b,b,,,,,]
        self.y = self.y.reshape(self.config.worker * self.config.sample, 12, 2).to(self.config.device)
        self.ref_ade = self.ref_ade.to(self.config.device)
        self.ref_fde = self.ref_fde.to(self.config.device)

        self.xt = torch.randn([self.config.worker * self.config.sample, 12, 2]).to(self.config.device)
        self.t = 100
        self.done = False

        return self.context, self.xt
    
    def step(self, x):
        self.xt = self.c0[self.t] * ( self.xt - self.c1[self.t] * x)
        reward = self.caculate_reward() if self.t == 1 else 0
        self.t -= 1

        return self.xt, reward
    
    def caculate_ade_fde(self):
        predicted_y_pos_rel = self.xt.cumsum(dim=1) * 0.4

        ade = (((predicted_y_pos_rel - self.y) ** 2).sum(-1) ** 0.5).mean(-1)
        fde = ((predicted_y_pos_rel[:, -1] - self.y[:, -1]) ** 2).sum(-1) ** 0.5

        ade = ade.reshape(self.config.worker , self.config.sample).min(-1)[0].mean()
        fde = fde.reshape(self.config.worker , self.config.sample).min(-1)[0].mean()

        return ade, fde
    
    def caculate_reward(self):
        predicted_y_pos_rel = self.xt.cumsum(dim=1) * 0.4

        ade = (((predicted_y_pos_rel - self.y) ** 2).sum(-1) ** 0.5).mean(-1)
        fde = ((predicted_y_pos_rel[:,  -1] - self.y[:,  -1]) ** 2).sum(-1) ** 0.5


        if self.config.reward_name == 'linear':
            reward = -ade


        elif self.config.reward_name == 'average_baseline_linear_10':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 10-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_10_negative_0.05':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 10-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.05
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_10_negative_0.1':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 10-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.1
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_10_negative_0.2':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 10-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.2
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_10_negative_0.4':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 10-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.4
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_10_negative_1':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 10-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 1
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_10_logadevar_1':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 10-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            
            adevar = ade.var(1).log().reshape(-1, 1).repeat(1, 20)
            adevar = (adevar - adevar.mean()) / adevar.var()
            reward += adevar

            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_10_logadevar_0.1':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 10-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            
            adevar = ade.var(1).log().reshape(-1, 1).repeat(1, 20)
            adevar = (adevar - adevar.mean()) / adevar.var()
            reward += 0.1 * adevar

            reward = reward.reshape(self.config.worker * self.config.sample)


        elif self.config.reward_name == 'average_ref_baseline_linear_10':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ref_ade, _ = self.ref_ade.sort(dim=1)
            sorted_ade, _ = ade.sort(dim=1)
            clip1 = ade <= 2 * sorted_ref_ade[:, 0].unsqueeze(1).repeat(1,20)#与best ref_ade距离在2倍以内的clip=True
            clip2 = ade <= sorted_ade[:, 3 - 1].unsqueeze(1).repeat(1,20)#当前最好3条轨迹为True
            clip3 = ade <= sorted_ade[:, 10 - 1].unsqueeze(1).repeat(1,20)#当前最好10条轨迹为True
            clip = (clip1 + clip2) * clip3
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_5_negative_0.1':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 5-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.1
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_8_negative_0.1':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 8-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.1
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_12_negative_0.1':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 12-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.1
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_15_negative_0.1':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 15-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.1
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_20_negative_0.1':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 20-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.1
            reward = reward.reshape(self.config.worker * self.config.sample)



        elif self.config.reward_name == 'average_baseline_linear_5_negative_0':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 5-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_5_negative_0.05':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 5-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.05
            reward = reward.reshape(self.config.worker * self.config.sample)


        elif self.config.reward_name == 'average_baseline_linear_5_negative_0.2':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 5-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.2
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_5_negative_0.5':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 5-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.5
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'onehot_1':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 0].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip

            reward =  clip.float() * 1
            reward -= clip.logical_not().float() * 0.1
            reward /= reward.std(unbiased=False)
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'onehot_3':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 3].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip

            reward =  clip.float() * 1
            reward -= clip.logical_not().float() * 0.1
            reward /= reward.std(unbiased=False)
            reward = reward.reshape(self.config.worker * self.config.sample)


        elif self.config.reward_name == 'average_baseline_fde_linear_10':
            fde = fde.reshape(self.config.worker , self.config.sample)

            sorted_fde, _ = fde.sort(dim=1)
            clip = fde <= sorted_fde[:, 10-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((fde * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -fde
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_fde_linear_5':
            fde = fde.reshape(self.config.worker , self.config.sample)

            sorted_fde, _ = fde.sort(dim=1)
            clip = fde <= sorted_fde[:, 5-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((fde * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -fde
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_ade_fde_linear_10':
            metric = fde + ade
            metric = metric.reshape(self.config.worker , self.config.sample)

            sorted_metric, _ = metric.sort(dim=1)
            clip = metric <= sorted_metric[:, 10-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((metric * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -metric
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_ade_fde_linear_5':
            metric = fde + ade
            metric = metric.reshape(self.config.worker , self.config.sample)

            sorted_metric, _ = metric.sort(dim=1)
            clip = metric <= sorted_metric[:, 5-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((metric * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -metric
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward = reward.reshape(self.config.worker * self.config.sample)


        elif self.config.reward_name == 'average_baseline_ade_fde_linear_10_negative_0.1':
            metric = fde + ade
            metric = metric.reshape(self.config.worker , self.config.sample)

            sorted_metric, _ = metric.sort(dim=1)
            clip = metric <= sorted_metric[:, 10-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((metric * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -metric
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not().float() * 0.1
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_ade_fde_linear_5_negative_0.1':
            metric = fde + ade
            metric = metric.reshape(self.config.worker , self.config.sample)

            sorted_metric, _ = metric.sort(dim=1)
            clip = metric <= sorted_metric[:, 5-1].unsqueeze(1).repeat(1,20)#选择ade最小的10次sample不clip
            baseline = ((metric * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,20)
  
            reward = -metric
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not().float() * 0.1
            reward = reward.reshape(self.config.worker * self.config.sample)

        elif self.config.reward_name == 'average_baseline_linear_20_negative_0.1_sample_80':
            ade = ade.reshape(self.config.worker , self.config.sample)

            sorted_ade, _ = ade.sort(dim=1)
            clip = ade <= sorted_ade[:, 20-1].unsqueeze(1).repeat(1,80)#选择ade最小的10次sample不clip
            baseline = ((ade * clip).sum(dim=1) / clip.sum(dim=1)).unsqueeze(1).repeat(1,80)
  
            reward = -ade
            baseline_reward = -baseline
            reward =  clip * (reward - baseline_reward)
            reward /= reward.std(unbiased=False)
            reward -= clip.logical_not() * 0.1
            reward = reward.reshape(self.config.worker * self.config.sample)

        else:
            print('reward_name error')
        
        now_ade = ade.min(-1)[0]
        self.now_ade[self.index] = now_ade

        with open(os.path.join(self.config.log_dir, 'log'), 'a') as f:
            f.write('now_ade_mean:{:.4f}\n'.format(self.now_ade.mean()))
            
        return reward





