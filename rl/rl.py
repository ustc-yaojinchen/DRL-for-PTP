import os
import torch
import dill
import numpy as np
from tqdm import tqdm

from dataset import get_timesteps_data
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers

from rl.dataset import BaseDataset
from rl.agent import Agent


class RL():
    def __init__(self, config):
        self.config = config

        self.hyperparams = get_traj_hypers()
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim//2
        # registar
        model_dir = os.path.join("./experiments", self.config.exp_name)
        registrar = ModelRegistrar(model_dir, "cpu")
        checkpoint = torch.load(os.path.join(model_dir, f"{self.config.dataset}_epoch{self.config.eval_at}.pt"), map_location = "cpu")
        registrar.load_models(checkpoint['encoder'])

        train_data_path = os.path.join(self.config.data_dir,self.config.dataset + "_train.pkl")
        eval_data_path = os.path.join(self.config.data_dir,self.config.dataset + "_test.pkl")
        with open(train_data_path, 'rb') as f:
            self.train_env = dill.load(f, encoding='latin1')
        with open(eval_data_path, 'rb') as f:
            self.eval_env = dill.load(f, encoding='latin1')

        encoder = Trajectron(registrar, self.hyperparams, "cpu")
        encoder.set_environment(self.train_env)
        encoder.set_annealing_params()
        self.model = AutoEncoder(self.config, encoder = encoder)
        self.model.load_state_dict(checkpoint['ddpm'])

        if self.config.pre:
            self.model.load_state_dict(torch.load(self.config.pre, map_location = "cpu"))

    @torch.no_grad()
    def get_data(self, mode):
        if mode == 'eval':
            env = self.eval_env
        elif mode == 'train':
            env = self.train_env

        model = self.model.to(self.config.device)
        context_all, future_rel_all, ade_all, fde_all = [], [], [], []
        for i, scene in enumerate(env.scenes):
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                batch = get_timesteps_data(env=env, scene=scene, t=timesteps, node_type="PEDESTRIAN", state=self.hyperparams['state'],
                               pred_state=self.hyperparams['pred_state'], edge_types=env.get_edge_types(),
                               min_ht=7, max_ht=self.hyperparams['maximum_history_length'], min_ft=12,
                               max_ft=12, hyperparams=self.hyperparams)
                if batch is None:
                    continue
                test_batch, nodes, timesteps_o = batch

                context = model.encoder.get_latent(test_batch, "PEDESTRIAN")
                predicted_y_vel = model.diffusion.sample(12, context.to(self.config.device), 20, bestof=True).detach().cpu()
                predicted_y_pos = predicted_y_vel.cumsum(dim=2) * 0.4

                for i in range(len(nodes)):
                    node, t = nodes[i], timesteps_o[i]
                    future_rel = node.get(np.array([t + 1, t + 12]), {'position': ['x', 'y']}) - node.get(np.array([t, t]), {'position': ['x', 'y']})
                    future_rel = torch.tensor(future_rel).float().unsqueeze(0)
                    context_i = context[i].float().unsqueeze(0).detach().cpu()
                    
                    error_i = predicted_y_pos[:, i] - future_rel.repeat(20,1,1)
                    ade_i = ((error_i ** 2).sum(-1) ** 0.5).mean(-1).unsqueeze(0)
                    fde_i = ((error_i[:, -1] ** 2).sum(-1) ** 0.5).unsqueeze(0)

                    future_rel_all.append(future_rel)
                    context_all.append(context_i)
                    ade_all.append(ade_i)
                    fde_all.append(fde_i)
                    print(ade_i.min(), fde_i.min())

        future_rel_all = torch.cat(future_rel_all)
        context_all = torch.cat(context_all)
        ade_all = torch.cat(ade_all)
        fde_all = torch.cat(fde_all)
        dataset = BaseDataset(context_all, future_rel_all, ade_all, fde_all)

        os.makedirs('rl/data', exist_ok=True)
        with open('rl/data/{:}-{:}.pkl'.format(self.config.dataset, mode), 'wb') as f:
            dill.dump(dataset, f)


    def train(self, rl_config):
        agent = Agent(self.model, config=rl_config)
        agent.test()
        for epoch in tqdm(range(rl_config.epochs)):
            agent.train_actions()
            agent.caculate_buffer()
            agent.update()
            agent.sigmas *= rl_config.discount_sigma
            agent.config.ceof_ref *= rl_config.discount_ref
            
            if epoch % 5 == 0:
                agent.test()
                torch.save(agent.model.state_dict(), os.path.join(agent.config.log_dir, "epoch_{:}.pt".format(epoch)))
                pass
