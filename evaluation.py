import torch
import torch, torchvision
import numpy as np
from PIL import Image


def get_img_from_obs(obs, camera_names, device):
    img_cameras = {}
    for camera in camera_names:

        
        pixel_obs = obs[f'{camera}_image'][::-1]
        img_cameras[camera] = torch.tensor(pixel_obs.copy()).float().permute(2, 0, 1).unsqueeze(0).to(device)/255
    return img_cameras

@torch.no_grad()
def run_eval(env, policy, device='cpu', num_episodes=1, gif_path=None):
    policy.eval()
    rewards = []
    
    episodes = []
    success_count = 0
    for _ in range(num_episodes):
        done = False
        episode_reward = 0
        obs = env.reset()
        imgs = []
        
        while not done:
            obs_img = get_img_from_obs(obs, env.camera_names, device)
            action = policy(obs_img).squeeze(0).cpu().numpy()
            obs, reward, done, _ = env.step(action)

            episode_reward += reward
            if gif_path is not None:
                imgs.append(torch.cat(list(obs_img.values()), dim=-1)[0].permute(1,2,0))
        if gif_path is not None:
            episodes.append(torch.stack(imgs))

        if episode_reward > 0:
            success_count += 1

        rewards.append(episode_reward)

    success_rate = success_count/num_episodes
    avg_reward = sum(rewards) / len(rewards)
    if gif_path is not None:
        episodes = np.array(torch.cat(episodes, dim=1).cpu().numpy()*255, dtype=np.uint8) #.permute()
        episodes = [Image.fromarray(frame) for frame in episodes][::4]
        episodes[0].save(gif_path.replace('.gif', f'_rw_{avg_reward}_sr_{success_rate}.gif') , save_all=True, append_images=episodes[1:], optimize=False, duration=200, loop=0)

    # episodes[0].save('outputs.gif', save_all=True, append_images=episodes[1:], optimize=False, duration=200, loop=0)
    # breakpoint()
    policy.train()

    # print(action.min(), action.max())
    return avg_reward, success_rate
