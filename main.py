import argparse
import torch
from typing import Optional
from policy import BcPolicy
# from evaluation import evaluate
from dataset import RobomimicDataset


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--episodes', type=int, default=None)
args = parser.parse_args()

# breakpoint()
cameras=['agentview', 'robot0_eye_in_hand']
dataset = RobomimicDataset('data/robomimic/square/processed_data96.hdf5', cameras=cameras, num_episode=args.episodes)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

# test_loader = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy = BcPolicy(action_dim=7, cameras=cameras).to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)


for epoch in range(args.epochs):
    
    for obs, action in train_loader:
        # breakpoint()
        obs = {k: v.to(device) for k, v in obs.items()}
        # obs = {k, v }
        loss = policy(obs, action.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)

    # scores = evaluate(eval_policy, dataset, seed=seed, num_game=cfg.num_eval_episode)


















