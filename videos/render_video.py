import numpy as np
import os
import imageio

file = "two_plane_video7.npy"
frames = np.load(file) 

imageio.mimsave(
    "two_last_iteration.mp4",
    frames,
    fps=120
)