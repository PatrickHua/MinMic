import torch
import torch.nn as nn
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
    def __init__(self, action_dim, cameras):
        super().__init__()

        self.encoders = nn.ModuleList([torchvision.models.resnet18(weights=None, num_classes=action_dim) for _ in cameras])

        self.policy = build_fc(
            in_dim=action_dim * len(cameras),
            hidden_dim=action_dim * 2,
            action_dim=action_dim,  # output dim
            num_layer=2,  # maybe more than 2?
            layer_norm=True,
            dropout=0,
        )
        self.cameras = cameras

    # def sample_action(self, obs):
    #     """TODO: diffusion policy sampling"""
    #     h = self.encoder(obs)
    #     pred_action = self.policy(h)  # policy contains tanh
    #     return pred_action

    def forward(self, obs: dict[str, torch.Tensor], action: Optional[torch.Tensor]=None):
        
        # forward encoder
        hs = []
        for i, camera in enumerate(self.cameras):
            x = obs[camera]
            h = self.encoders[i](x)
            hs.append(h)

        h = torch.cat(hs, dim=1)
        pred_action = self.policy(h)  # policy contains tanh
        
        if action is None:
            return pred_action
        else:
            loss = nn.functional.mse_loss(pred_action, action)        
            return loss


    # @torch.no_grad()
    # def act(self, obs: dict[str, torch.Tensor]):
    #     """TODO: delete this?"""
    #     assert not self.training  # model.eval() on

    #     greedy_action = self.sample_action(obs)
    #     return greedy_action  #.detach().cpu()


if __name__ == '__main__':
    action_dim = 10
    policy = BcPolicy(action_dim=action_dim, cameras=['agentview', 'egoview'])
    obs = dict(agentview=torch.zeros((1, 3, 224, 224)), egoview=torch.zeros((1, 3, 224, 224)))
    action = torch.zeros((1, action_dim))
    loss = policy(obs, action)
    print(loss)
    