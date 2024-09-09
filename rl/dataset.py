from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, encode, y, ade ,fde):
        super(Dataset, self).__init__()
        self.encode = encode
        self.y = y
        self.ade = ade
        self.fde = fde

    def __len__(self):
        return self.encode.shape[0]
    
    def __getitem__(self, index):
        return self.encode[index], self.y[index], self.ade[index], self.fde[index]


class PPODataset(Dataset):
    def __init__(self, agent):
        super(Dataset, self).__init__()
        self.contexts   = agent.contexts.reshape(100 * agent.config.worker * agent.config.sample, 256)
        self.xts        = agent.xts.reshape(100 * agent.config.worker * agent.config.sample, 12, 2)
        self.betas      = agent.betas.reshape(100 * agent.config.worker * agent.config.sample)

        self.mus        = agent.mus.reshape(100 * agent.config.worker * agent.config.sample, 12, 2)
        self.sigmas     = agent.sigmas.reshape(100 * agent.config.worker * agent.config.sample, 12, 2)
        self.actions    = agent.actions.reshape(100 * agent.config.worker * agent.config.sample, 12, 2)
        self.advantages = agent.advantages.reshape(100 * agent.config.worker * agent.config.sample)

        # self.advantages = (self.advantages - self.advantages.mean()) / self.advantages.std(unbiased=False)

    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, index):
        return self.contexts[index], self.xts[index], self.betas[index], self.mus[index], self.sigmas[index], self.actions[index], self.advantages[index]
