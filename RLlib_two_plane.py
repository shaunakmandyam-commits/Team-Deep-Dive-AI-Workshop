import os
import numpy as np
import ray
import torch
from pprint import pprint
from gymnasium.spaces import Box
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import PolicySpec
from TwoPlaneMultiAgent import TwoPlaneMultiAgent

path = os.path.abspath("models/fly_towards_airport_policy")
one_plane_algo = PPO.from_checkpoint(path)

config = PPOConfig().environment(env=TwoPlaneMultiAgent)

config.multi_agent(policies={"plane_policy" : PolicySpec()},
                   policy_mapping_fn=lambda agent_id, *args, **kwargs: "plane_policy")
config.training(lr=1e-4,
                entropy_coeff=0.001,
                train_batch_size=2048,
                minibatch_size=64,
                num_epochs=15,
                model={
                    "fcnet_hiddens": [128,128],
                    "fcnet_activation": "tanh"})
config.env_runners(num_env_runners=1)


config.callbacks()




two_plane_algo = config.build()

from transfer_RL import transfer_weights


"""weights = transfer_weights("plane_policy", one_plane_algo, two_plane_algo)
final_weights = {"plane_policy": weights}
two_plane_algo.set_weights(final_weights)
"""

path = os.path.abspath("models\separate_plane_policy")
two_plane_algo = PPO.from_checkpoint(path)

print("model loaded")

for i in range(1000):
    result = two_plane_algo.train()
    if i % 100 == 0 and i != 0:
        checkpoint = two_plane_algo.save(path)
        print("Saved:", checkpoint)    
    #pprint(result)
    print(
        "Iteration:", result["training_iteration"], "\n",
        "Reward:", result["env_runners"]["episode_return_mean"], "\n",
        "Episode Length:", result["env_runners"]["episode_len_mean"], "\n",
        "Best Reward:", result["env_runners"]["episode_return_max"], "\n",
        "Min Ep Length:", result["env_runners"]["episode_len_min"], "\n",
        "Episodes:", result["env_runners"]["num_episodes"], "\n",
        "VF Explained:", result["learners"]["plane_policy"]["vf_explained_var"], "\n",
        "Policy Loss:", result["learners"]["plane_policy"]["policy_loss"], "\n",
        "Value Loss:", result["learners"]["plane_policy"]["vf_loss"], "\n",
        "KL:", result["learners"]["plane_policy"]["mean_kl_loss"], "\n",
        "Entropy:", result["learners"]["plane_policy"]["entropy"], "\n"
    )