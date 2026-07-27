import ray
import numpy as np

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from OnePlaneMultiAgent import OnePlaneMultiAgent
from OnePlaneCallbacks import OnePlaneCallbacks
from pprint import pprint
from save_simulation import sim_human, sim_rgb


new_model = False


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
    lr=3e-4,
    train_batch_size=2048,
    minibatch_size=64,
    num_epochs=15,
    model={
        "fcnet_hiddens": [128,128],
        "fcnet_activation": "tanh",
    }
)

config.env_runners(num_env_runners=1, create_env_on_local_worker=True)
#config.callbacks(callbacks_class=OnePlaneCallbacks)


algo = config.build()

import os
path = os.path.abspath("models/fly_towards_airport_policy")

if not new_model:
    algo.restore_from_path(path)
    print("model loaded")



for i in range(2000):
    result = algo.train()
    env : OnePlaneMultiAgent = algo.env_runner.env.envs[0].env
    if i % 100 == 0:
        if i != 0:
            checkpoint = algo.save(path)
            print("Saved:", checkpoint) 

        video = sim_rgb(OnePlaneMultiAgent, algo, episodes=1, environment=env)
        folder = "videos"
        file = "one_plane_video" + str(int(result['training_iteration'] / 100) )
        full_path = os.path.join(folder, file)
        np.save(full_path, video)

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