import torch


def get_img_from_obs(obs, camera_names, device):
    img_cameras = {}
    for camera in camera_names:
        pixel_obs = obs[f'{camera}_image'][::-1]
        img_cameras[camera] = torch.tensor(pixel_obs.copy()).float().permute(2, 0, 1).unsqueeze(0).to(device)
    return img_cameras

@torch.no_grad()
def run_eval(env, policy, device='cpu', num_episodes=1):

    rewards = []
    for _ in range(num_episodes):
        done = False
        total_reward = 0
        obs = env.reset()
        obs_img = get_img_from_obs(obs, env.camera_names, device)
        x = 0
        while not done:
            # breakpoint()
            # print(x:=x+1)
            action = policy(obs_img).squeeze(0).cpu().numpy()
            # try:
            obs, reward, done, _ = env.step(action)
            # except Exception:
                # breakpoint()
            total_reward += reward
            obs_img = get_img_from_obs(obs, env.camera_names, device)

        rewards.append(total_reward)

    return sum(rewards) / len(rewards)
