import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC
from .rl_utils import (
    compute_attribution,
    compute_attribution_mask,
    compute_attribution_mask_bymean
)
import random

class SDA(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.args = args
        self.quantile = args.sda_quantile
        self.attrib_quantile = 0.95
        
        self.normal_obs_predictor = m.ImagePredictor(self.critic.encoder).to(self.device)
        self.recon_optimizer = torch.optim.Adam(
            self.normal_obs_predictor.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
        )

        self.attribution_predictor = m.AttributionPredictor(action_shape[0],self.critic.encoder).to(self.device)
        self.aux_update_freq = args.aux_update_freq
        self.aux_optimizer = torch.optim.Adam(
            self.attribution_predictor.parameters(),
            lr=args.aux_lr,
            betas=(args.aux_beta, 0.999),
        )
    
    def select_action(self, obs, eval_mode='normal'):
        _obs = self._obs_to_input(obs)
        #if 'video' in eval_mode:
        #    # reconstruct the normal obs
        #    _obs = self.normal_obs_predictor(_obs.detach())
        with torch.no_grad():
            mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def salient_overlay(self, obs, action, step=None):
        # salient overlay
        obs_grad = compute_attribution(self.critic,obs,action.detach())
        mask = compute_attribution_mask(obs_grad,float(self.quantile))
        #salient_mask = compute_attribution_mask(obs_grad,float(self.attrib_quantile))
        salient_mask = compute_attribution_mask(obs_grad,float(self.quantile))

        # images from place365 dataset
        imgs = augmentations._get_places_batch(batch_size=obs.size(0)).repeat(1, obs.size(1) // 3, 1, 1)

        # imgs $\in$ (0,1), obs $\in$ (0, 255)
        salient_obs = obs * salient_mask
        salient_obs[salient_mask<1] = random.uniform(obs.view(-1).min(),obs.view(-1).max())
        masked_obs = obs * mask + imgs * (~mask) * 255.0
        overlay_obs = obs * 0.5 + imgs * 0.5 * 255.0
        #masked_overlay_obs = obs * mask + overlay_obs * (~mask)

        if step % self.args.save_freq == 0 or step==self.args.init_steps+1:
            # save obs, imgs, masked_obs
            self.masked_obs_path = os.path.join(self.work_dir, 'masked_obs') 
            if not os.path.exists(self.masked_obs_path):
                os.mkdir(self.masked_obs_path)
            np.save(os.path.join(self.masked_obs_path, 'obs_{}'.format(step)), obs.cpu().detach().numpy())
            np.save(os.path.join(self.masked_obs_path, 'img_{}'.format(step)), imgs.cpu().detach().numpy())
            np.save(os.path.join(self.masked_obs_path, 'masked_obs_{}'.format(step)), masked_obs.cpu().detach().numpy())
        
        return salient_obs, masked_obs, overlay_obs

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # salient obs (mask*obs), masked obs (mask*obs+(1-mask)*img), overlay obs (obs*0.5 + img*0.5)
        salient_obs, masked_obs, overlay_obs = self.salient_overlay(obs, action, step)

        # use [obs, overlay_obs, masked_obs] for training (sda_quantile0.9)
        total_obs = torch.cat([obs, overlay_obs, masked_obs], axis=0)
        total_action = torch.cat([action, action, action], axis=0)
        total_target_Q = torch.cat([target_Q, target_Q, target_Q], axis=0)
        current_Q1, current_Q2 = self.critic(total_obs, total_action)
        critic_loss = F.mse_loss(current_Q1, total_target_Q) + F.mse_loss(current_Q2, total_target_Q)
        
        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        recon_loss_numpy = None

        return critic_loss.detach().cpu().numpy(), recon_loss_numpy

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()

        critic_loss, recon_loss = self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        actor_loss, alpha_loss, alpha_value = None, None, None
        if step % self.actor_update_freq == 0:
            actor_loss, alpha_loss, alpha_value = self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        return critic_loss, recon_loss, actor_loss, alpha_loss, alpha_value