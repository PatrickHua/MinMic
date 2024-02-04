import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from encoder import MultiViewEncoder
import torchvision
from typing import Optional

def build_fc(in_dim, hidden_dim, action_dim, num_layer, layer_norm, dropout):
    dims = [in_dim]
    dims.extend([hidden_dim for _ in range(num_layer)])

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if layer_norm == 2 and (i == num_layer - 1):
            layers.append(nn.LayerNorm(dims[i + 1]))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-1], action_dim))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class BcPolicy(nn.Module):
    def __init__(self, action_dim, cameras, diffusion=False):
        super().__init__()

        self.encoders = nn.ModuleList([torchvision.models.resnet18(weights=None, num_classes=action_dim) for _ in cameras])

        self.policy = build_fc(
            in_dim=action_dim * (len(cameras)+1) + 1 if diffusion else action_dim * len(cameras),
            hidden_dim=action_dim * 2,
            action_dim=action_dim,  # output dim
            num_layer=2,  # maybe more than 2?
            layer_norm=True,
            dropout=0,
        )
        self.action_dim = action_dim
        self.cameras = cameras
        self.diffusion = diffusion

    def sample_action(self, h, nb_step=10):
        """diffusion policy sampling"""
        if self.diffusion:
            bz = h.shape[0]
            noisy_action = torch.randn((bz, self.action_dim))

            for t in range(nb_step):
                inp = torch.cat([h, noisy_action, torch.tensor(t/nb_step).to(h).view(1, 1).repeat(bz, 1)], dim=-1)
                tangent = self.policy(inp)
                noisy_action = noisy_action + tangent / nb_step
            return noisy_action
        else:
            pred_action = self.policy(h)
        return pred_action

    def policy_loss(self, h, action):
        if self.diffusion:
            alpha = torch.rand(h.shape[0], device=h.device).unsqueeze(-1)
            noise = torch.randn_like(action)
            noisy_action = alpha * action + (1 - alpha) * noise
            
            tangent = self.policy(torch.cat([h, noisy_action, alpha], dim=-1))
            loss = F.mse_loss(tangent, action - noise)
        else:
            pred_action = self.policy(h)
            loss = F.mse_loss(pred_action, action)

        return loss


    def forward(self, data: dict[str, torch.Tensor]):
        # forward encoder
        hs = []
        for i, camera in enumerate(self.cameras):
            x = data[camera]
            h = self.encoders[i](x)
            hs.append(h)

        h = torch.cat(hs, dim=1)
        # pred_action = self.policy(h)  # policy contains tanh

        if data.get('action') is None:
            return self.sample_action(h)
        else:
            return self.policy_loss(h, data['action'])


    # @torch.no_grad()
    # def act(self, obs: dict[str, torch.Tensor]):
    #     """TODO: delete this?"""
    #     assert not self.training  # model.eval() on

    #     greedy_action = self.sample_action(obs)
    #     return greedy_action  #.detach().cpu()


if __name__ == '__main__':
    action_dim = 10
    policy = BcPolicy(action_dim=action_dim, cameras=['agentview', 'egoview'], diffusion=True)
    obs = dict(agentview=torch.zeros((1, 3, 224, 224)), egoview=torch.zeros((1, 3, 224, 224)), action=torch.zeros((1, action_dim)))
    # action = 
    loss = policy(obs)
    # policy.sample_action()
    obs.pop('action')
    sampled = policy(obs)
    print(loss, sampled.shape)
    