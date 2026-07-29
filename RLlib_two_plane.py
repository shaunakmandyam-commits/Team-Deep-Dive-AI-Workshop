import os
import numpy as np
import ray
import torch
from pprint import pprint
from gymnasium.spaces import Box
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import PolicySpec
from TwoPlaneMultiAgent import TwoPlaneMultiAgent
from save_simulation import sim_human, sim_rgb


new_model = True

path = os.path.abspath("models/fly_towards_airport_policy")
one_plane_algo = PPO.from_checkpoint(path)

from torch.utils.tensorboard import SummaryWriter
tb_log_dir = os.path.abspath("./my_tensorboard_logs/two_plane_logs")
writer = SummaryWriter(log_dir=tb_log_dir)


config = PPOConfig().environment(env=TwoPlaneMultiAgent)
config.multi_agent(policies={"plane_policy" : PolicySpec()},
                   policy_mapping_fn=lambda agent_id, *args, **kwargs: "plane_policy")
config.training(lr=1e-5,
                entropy_coeff=0.001,
                train_batch_size=4096,
                minibatch_size=64,
                num_epochs=10,
                model={
                    "fcnet_hiddens": [128,128],
                    "fcnet_activation": "tanh"})
config.env_runners(num_env_runners=2, create_env_on_local_worker=True, sample_timeout_s=120)

two_plane_algo = config.build()

from transfer_RL import transfer_weights

path = os.path.abspath("models\separate_plane_policy")

if new_model:
    weights = transfer_weights("plane_policy", one_plane_algo, two_plane_algo)
    final_weights = {"plane_policy": weights}
    two_plane_algo.set_weights(final_weights)
    two_plane_algo.save(path)
    print('saved new model')
else:
    two_plane_algo = PPO.from_checkpoint(path)
    print("model loaded")

for i in range(10000):
    result = two_plane_algo.train()

    iteration = result['training_iteration']
    env : TwoPlaneMultiAgent = two_plane_algo.env_runner.env.envs[0].env
    if i % 100 == 0:
        if i != 0:
            checkpoint = two_plane_algo.save(path)
            print("Saved:", checkpoint) 

        video = sim_rgb(TwoPlaneMultiAgent, two_plane_algo, episodes=10, difficulty=env.difficulty)
        folder = "videos"
        file = "two_plane_video" + str(int(iteration / 100) )
        full_path = os.path.join(folder, file)
        np.save(full_path, video)
        print("video saved in", file)

    """
    mean ep length
    mean ep rew
    entropy
    policy loss
    value function loss
    value function explained variance
    policy loss
    kl loss
    """
    metrics = {"Reward" : result["env_runners"]["episode_return_mean"], 
               "Episode Length" : result["env_runners"]["episode_len_mean"], 
               "Best Reward" : result["env_runners"]["episode_return_max"], 
               "Min Ep Length" : result["env_runners"]["episode_len_min"], 
               "Episodes" : result["env_runners"]["num_episodes"], 
               "VF Explained" : result["learners"]["plane_policy"]["vf_explained_var"],
               "Policy Loss:" : result["learners"]["plane_policy"]["policy_loss"], 
               "Value Loss" : result["learners"]["plane_policy"]["vf_loss"], 
               "KL" : result["learners"]["plane_policy"]["mean_kl_loss"], 
               "Entropy" : result["learners"]["plane_policy"]["entropy"]}

    print("iteration :", iteration)
    for key, val in metrics.items():
        print(key, ":", val)
        writer.add_scalar(key, val, iteration)
    print()