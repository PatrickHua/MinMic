import os
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
parser.add_argument('--output_dir', type=str, default='outputs/')
parser.add_argument('--eval_interval', type=int, default=1, help='Evaluate every x epochs')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

dataset = RobomimicDataset('data/robomimic/square/processed_data96.hdf5', cameras=args.cameras, num_episode=args.episodes)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

# test_loader = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy = BcPolicy(action_dim=7, cameras=args.cameras).to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)


for epoch in range(args.epochs):
    
    
    losses = []
    for data in train_loader:

        loss = policy({k: v.to(device) for k, v in data.items()})
        policy.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # print(f'{loss.item():.4f}')
    if epoch % args.eval_interval == 0:
        avg_loss = sum(losses)/len(losses)
        score, success_rate = run_eval(dataset.env, policy, device, num_episodes=args.eval_episodes, gif_path=os.path.join(args.output_dir, f'epoch_{epoch}_loss_{avg_loss:.4f}.gif'))
        print(f'Epoch {epoch} Score {score} Loss {avg_loss:.4f} Success rate {success_rate:.4f}')
    # episodes_viz
    # scores = evaluate(eval_policy, dataset, seed=seed, num_game=cfg.num_eval_episode)


















