import argparse
import torch
from typing import Optional
from policy import BcPolicy
# from evaluation import evaluate
from dataset import RobomimicDataset
from evaluation import run_eval


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--episodes', type=int, default=None)
parser.add_argument('--cameras', nargs='+', default=['agentview', 'robot0_eye_in_hand'], help='Which view is needed. Expects at least one camera')
parser.add_argument('--eval_episodes', type=int, default=2)

args = parser.parse_args()


dataset = RobomimicDataset('data/robomimic/square/processed_data96.hdf5', cameras=args.cameras, num_episode=args.episodes)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

# test_loader = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy = BcPolicy(action_dim=7, cameras=args.cameras).to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)


for epoch in range(args.epochs):
    
    score = run_eval(dataset.env, policy, device, num_episodes=args.eval_episodes)
    
    losses = []
    for obs, action in train_loader:

        obs = {k: v.to(device) for k, v in obs.items()}
        loss = policy(obs, action.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # print(f'{loss.item():.4f}')
    print(f'Epoch {epoch} Score {score} Loss {sum(losses)/len(losses):.4f}')

    # scores = evaluate(eval_policy, dataset, seed=seed, num_game=cfg.num_eval_episode)


















