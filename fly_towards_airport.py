import ray
import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from OnePlaneMultiAgent import OnePlaneMultiAgent
from OnePlaneCallbacks import OnePlaneCallbacks
from pprint import pprint
from save_simulation import sim_human, sim_rgb

import os
path = os.path.abspath("models/video_train")

from torch.utils.tensorboard import SummaryWriter
tb_log_dir = os.path.abspath("./my_tensorboard_logs/test_logs")
writer = SummaryWriter(log_dir=tb_log_dir)

new_model = True


config = (
    PPOConfig()
    .environment(env=OnePlaneMultiAgent)
    .multi_agent(
        policies={
            "plane_policy": PolicySpec()
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "plane_policy",
    )
)
config.training(
    lr=1e-4,
    entropy_coeff=0.001,
    train_batch_size=4096,
    minibatch_size=64,
    num_epochs=10,
    model={
        "fcnet_hiddens": [128,128],
        "fcnet_activation": "tanh",
    }
)

config.env_runners(num_env_runners=2, create_env_on_local_worker=True)



algo = config.build()


if not new_model:
    algo.restore_from_path(path)
    print("model loaded")



for i in range(1000):
    result = algo.train()

    iteration = result['training_iteration']
    env : OnePlaneMultiAgent = algo.env_runner.env.envs[0].env
    if i % 100 == 0:
        if i != 0:
            checkpoint = algo.save(path)
            print("Saved:", checkpoint) 

        video = sim_rgb(OnePlaneMultiAgent, algo, episodes=5, difficulty=env.difficulty)
        folder = "videos"
        file = "short_test" + str(int(iteration / 100) )
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