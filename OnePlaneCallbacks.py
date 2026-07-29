
from ray.rllib.callbacks.callbacks import RLlibCallback



class OnePlaneCallbacks(RLlibCallback):
    def on_train_result(self, *, algorithm, metrics_logger = None, result, **kwargs):

        print(type(algorithm))
        print(type(algorithm.env_runner))
        print(type(algorithm.env_runner_group))
        print(type(algorithm.env_runner.env))
