# First install:
#                   pip install moviepy
# Reference: https://www.anyscale.com/blog/an-introduction-to-reinforcement-learning-with-openai-gym-rllib-and-google#creating-a-video-of-the-trained-model-in-action

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# Create Pendulum environment
# render_mode = "human" # for display the video in real-time
# render_mode = "human" # for saving the video in files
def save_animation(env):
    # env = gym.make('Pendulum-v1', render_mode="rgb_array")
    after_training = "after_training.mp4"
    after_video = VideoRecorder(env, after_training)
    # Reset the environment to the initial state
    observation = env.reset()

done = False

for _ in range(120):  # Record 120 frames
    env.render()
    after_video.capture_frame()
    # Take a random action
    action = env.action_space.sample()
    observation, reward, done, info, _ = env.step(action)

after_video.close()
env.close()