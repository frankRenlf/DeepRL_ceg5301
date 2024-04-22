###########################################################################################
# Training of Deep Q-Learning Networks (DQN)
# Paper: https://www.nature.com/articles/nature14236
# Reference: https://github.com/Kchu/DeepRL_PyTorch
###########################################################################################
# The codes for the classical control game Pendulum are excerpted from OpenAI baseline Gym,
# and an new image data output API is added to this game by defining a new observation wrapper
# in wrappers.py.
# Parallel DQN for an Inverted Pendulum with Image Data implemented in PyTorch and OpenAI Gym
import os
import torch
import pickle
import time
import argparse
import numpy as np
from collections import deque

import dqn
from dqn import DQN
from wrappers import wrap, wrap_cover, wrap_cover_pendulun, wrap_cover_pendulun_test
from parallel import SubprocVecEnv
from pendulum import PendulumEnv
from gym.wrappers.monitoring.video_recorder import VideoRecorder


parser = argparse.ArgumentParser(description="Some settings of the experiment.")
parser.add_argument(
    "--game",
    type=str,
    nargs=1,
    default="Pendulum",
    help="name of the games. for example: Breakout",
)

parser.add_argument("--idling", help=" ", action="store_true", default=False)

args = parser.parse_args()
args.games = "".join(args.game)

# the discrete action space
disc_actions = [
    np.array(-2, dtype=np.float32),
    np.array(-1.5, dtype=np.float32),
    np.array(-1, dtype=np.float32),
    np.array(-0.5, dtype=np.float32),
    np.array(0, dtype=np.float32),
    np.array(0.5, dtype=np.float32),
    np.array(1, dtype=np.float32),
    np.array(1.5, dtype=np.float32),
    np.array(2, dtype=np.float32),
]
# dims = np.arange(-4, 4.5, 0.5, dtype=np.float32)
# disc_actions = [np.array(d, dtype=np.float32) for d in dims]


def main():
    """Environment Settings"""
    # number of environments for C51
    N_ENVS = 1
    # Total simulation step
    STEP_NUM = int(1.5e5)
    # visualize for agent playing
    RENDERING = False
    # openai gym env name
    # ENV_NAME = args.game+'NoFrameskip-v4'
    global args
    ENV_NAME = args.game
    # idling
    IDLING = args.idling

    """Training Settings"""
    # simulator steps for start learning
    LEARN_START = int(1e3)
    # simulator steps for learning interval
    LEARN_FREQ = 4
    # epsilon-greedy
    EPSILON = 1.0
    # the length pf each episode
    EPISODE_LENGTH = 200
    # create envs
    global disc_actions
    env = SubprocVecEnv(
        [
            wrap_cover_pendulun(ENV_NAME, disc_actions, episode_length=EPISODE_LENGTH)
            for i in range(N_ENVS)
        ]
    )

    """Save&Load Settings"""
    # check save/load
    SAVE = True
    LOAD = False
    BUFFER_LOAD = False
    # save frequency
    SAVE_FREQ = int(1e3) // N_ENVS
    # paths for predction net, target net, result log
    current_path = os.path.dirname(os.path.realpath("__file__"))
    PRED_PATH = os.path.join(
        current_path, "data/model/dqn_pred_net_o_" + args.games + ".pkl"
    )
    TARGET_PATH = os.path.join(
        current_path, "data/model/dqn_target_net_o_" + args.games + ".pkl"
    )
    RESULT_PATH = os.path.join(
        current_path, "data/plots/dqn_result_o_" + args.games + ".pkl"
    )
    BUFFER_PATH = os.path.join(
        current_path, "data/replay_buffer/dqn_buffer_" + args.games + ".pkl"
    )
    # create directory
    if not os.path.exists(os.path.dirname(RESULT_PATH)):
        os.makedirs(os.path.dirname(RESULT_PATH))
    if not os.path.exists(os.path.dirname(PRED_PATH)):
        os.makedirs(os.path.dirname(PRED_PATH))
    if not os.path.exists(os.path.dirname(TARGET_PATH)):
        os.makedirs(os.path.dirname(TARGET_PATH))
    if not os.path.exists(os.path.dirname(BUFFER_PATH)):
        os.makedirs(os.path.dirname(BUFFER_PATH))

    # define agent
    dqn = DQN()

    # model load with check
    if LOAD and os.path.isfile(PRED_PATH) and os.path.isfile(TARGET_PATH):
        dqn.load_model(PRED_PATH, TARGET_PATH)
        pkl_file = open(RESULT_PATH, "rb")
        result = pickle.load(pkl_file)
        pkl_file.close()
        print("Load complete!")
    else:
        result = []
        print("Initialize results!")

    # offline data load with check
    if BUFFER_LOAD and os.path.isfile(BUFFER_PATH):
        dqn.load_buffer(BUFFER_PATH)
        print("Load data complete!")

    print("Collecting experience...")

    # episode step for accumulate reward
    epinfobuf = deque(maxlen=100)
    # check learning time
    start_time = time.time()

    # env reset
    s = env.reset()
    # print(s.shape)
    mx = [-200]
    # for step in tqdm(range(1, STEP_NUM//N_ENVS+1)):
    for step in range(1, STEP_NUM // N_ENVS + 1):
        # When loading data from a file, don't need to sample from env.
        if not BUFFER_LOAD:
            a = dqn.choose_action(s, EPSILON, IDLING)

            # take action and get next state
            s_, r, done, infos, _ = env.step(a)
            if done.all():
                # log arrange
                for info in infos:
                    maybeepinfo = info.get("episode")
                    if maybeepinfo:
                        epinfobuf.append(maybeepinfo)
                s_ = env.reset()

            # store the transition
            for i in range(N_ENVS):
                dqn.store_transition(s[i], a[i], r[i], s_[i], done[i])

        # annealing the epsilon(exploration strategy)
        if step <= int(5e4 / N_ENVS):
            # linear annealing to 0.9 until million step
            EPSILON -= 0.9 / 5e4 * N_ENVS
        elif step <= int(1.5e5 / N_ENVS) and step > 5e4:
            # else:
            # linear annealing to 0.99 until the end
            EPSILON -= 0.099 / 1e5 * N_ENVS

        # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
        if (
            (not IDLING)
            and (LEARN_START <= dqn.memory_counter)
            and (dqn.memory_counter % LEARN_FREQ == 0)
        ):
            loss = dqn.learn()

        # print log and save
        if step % SAVE_FREQ == 0:
            # check time interval
            time_interval = round(time.time() - start_time, 2)
            # calc mean return (the average return over the last 100 episodes)
            mean_100_ep_return = round(
                np.mean([epinfo["r"] for epinfo in epinfobuf]), 2
            )
            result.append(mean_100_ep_return)
            # print log
            print(
                "Used Step: ",
                dqn.memory_counter,
                "| EPS: ",
                round(EPSILON, 3),
                # '| Loss: ', loss,
                "| Mean ep 100 return: ",
                mean_100_ep_return,
                "| Used Time:",
                time_interval,
            )
            # save model
            dqn.save_model(PRED_PATH, TARGET_PATH)
            pkl_file = open(RESULT_PATH, "wb")
            pickle.dump(np.array(result), pkl_file)
            pkl_file.close()
            # render
            # Once the performance is good enough (>-200), deploy the learned greedy policy
            if mean_100_ep_return >= -200 and mean_100_ep_return >= mx[0]:
                # show the animation and record the mean_100_ep_return (the average return over the last 100 episodes)
                evaluate_performance(dqn, disc_actions, episode_length=EPISODE_LENGTH)
                # re-simulate to save the video
                save_animation(dqn, disc_actions)
                mx[0] = mean_100_ep_return
                break

        if not BUFFER_LOAD:
            s = list(s_)
            if RENDERING:
                env.render()

    if not BUFFER_LOAD:
        dqn.save_buffer(BUFFER_PATH)
    print("The training is done!")


def evaluate_performance(
    agent, disc_actions, render_mode="human", episode_length=200, test_num=1
):
    # create env
    global args
    game = args.game
    env = wrap_cover_pendulun_test(game, disc_actions, render_mode, episode_length)()

    for _ in range(test_num):
        # initialize the env
        s, _ = env.reset()
        s = [s]
        # render
        env.render()

        for step in range(episode_length):
            a = agent.choose_action(s, 0.0, idling=False)
            s_, r, done, infos, _ = env.step(a[0])
            s = [s_]

            env.render()
    env.close()


def save_animation(agent, disc_actions, render_mode="rgb_array", frames_length=200):
    # create env
    # render_mode = "human" # for display the video in real-time
    # render_mode = "human" # for saving the video in files
    global args
    game = args.game
    env = wrap_cover_pendulun_test(game, disc_actions, render_mode, frames_length)()

    after_training = "after_training.mp4"
    after_video = VideoRecorder(env, after_training)
    # initialize the env
    s, _ = env.reset()
    s = [s]
    # render
    env.render()

    for step in range(frames_length):
        after_video.capture_frame()
        a = agent.choose_action(s, 0.0, idling=False)
        s_, r, done, infos, _ = env.step(a[0])
        s = [s_]
        env.render()

    after_video.close()
    env.close()


def get_video():
    current_path = os.path.dirname(os.path.realpath("__file__"))
    PRED_PATH = os.path.join(
        current_path, "data/model/dqn_pred_net_o_" + args.games + ".pkl"
    )
    TARGET_PATH = os.path.join(
        current_path, "data/model/dqn_target_net_o_" + args.games + ".pkl"
    )
    dqn = DQN()
    dqn.load_model(PRED_PATH, TARGET_PATH)
    evaluate_performance(dqn, disc_actions)
    save_animation(dqn, disc_actions)


if __name__ == "__main__":
    # main()
    get_video()
