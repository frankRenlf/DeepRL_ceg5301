###########################################################################################
# Implementation of Deep Q-Learning Networks (DQN)
# Paper: https://www.nature.com/articles/nature14236
# Reference: https://github.com/Kchu/DeepRL_PyTorch
###########################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from replay_memory import ReplayBufferImage


"""DQN settings"""
# sequential images to define state
STATE_LEN = 4
# target policy sync interval
TARGET_REPLACE_ITER = 2
# (prioritized) experience replay memory size
MEMORY_CAPACITY = int(1e5)
# gamma for MDP
GAMMA = 0.99

"""Training settings"""
# check GPU usage
USE_GPU = (
    True if torch.cuda.is_available() or torch.backends.mps.is_available() else False
)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
print("USE GPU: " + str(USE_GPU) + " " + str(DEVICE))
# mini-batch size
BATCH_SIZE = 64
# learning rate
LR = 2e-4
# the number of actions
N_ACTIONS = 9
# the dimension of states
N_STATE = 4
# the multiple of tiling states
N_TILE = 20

input_actions = 1
input_states = 0


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.feature_extraction = nn.Sequential(
            # Conv2d(input channels, output channels, kernel_size, stride)
            nn.Conv2d(STATE_LEN, 8, kernel_size=8, stride=2),
            # TODO: ADD SUITABLE CNN LAYERS TO ACHIEVE BETTER PERFORMANCE
            nn.ReLU(),
            nn.Conv2d(8, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # action value
        self.fc_q = nn.Linear(8 * 8 * 8 + N_TILE * N_STATE, N_ACTIONS)
        self.fc_q1 = nn.Linear(N_ACTIONS, 1)

        # TODO: ADD SUITABLE FULLY CONNECTED LAYERS TO ACHIEVE BETTER PERFORMANCE
        self.fc_0 = nn.Linear(
            512 + N_TILE * N_STATE * input_states + N_TILE * input_actions,
            512,
        )
        self.fc_1 = nn.Linear(512, 8 * 8 * 8 + N_TILE * N_STATE)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, state=None, action=None):
        # x.size(0) : minibatch size
        # print(x.shape, state.shape)
        mb_size = x.size(0)
        x = self.feature_extraction(x / 255.0)  # (m, 9 * 9 * 10)
        x = x.view(x.size(0), -1)
        if input_states > 0:
            state = state.view(state.size(0), -1)
            state = torch.tile(state, (1, N_TILE))
            x = torch.cat((x, state), 1)
        if input_actions > 0:
            action = action.view(action.size(0), -1)
            action = torch.tile(action, (1, N_TILE))
            x = torch.cat((x, action), 1)

        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_1(x))
        action_value = self.fc_q(x)
        if input_actions > 0:
            action_value = self.fc_q1(action_value)
        return action_value

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(
            torch.load(
                path,
                map_location=(
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                ),
            ),
        )


class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet(), ConvNet()
        # sync target net
        self.update_target(self.target_net, self.pred_net, 1.0)
        # use gpu
        if USE_GPU:
            self.pred_net = self.pred_net.to(DEVICE)
            self.target_net = self.target_net.to(DEVICE)

        # simulator step counter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0
        # loss function
        self.loss_function = nn.MSELoss()
        # create the replay buffer
        self.replay_buffer = ReplayBufferImage(MEMORY_CAPACITY)

        # define optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)

    def update_target(self, target, pred, update_rate):
        # update target network parameters using prediction network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_(
                (1.0 - update_rate) * target_param.data + update_rate * pred_param.data
            )

    def save_model(self, pred_path, target_path):
        # save prediction network and target network
        self.pred_net.save(pred_path)
        self.target_net.save(target_path)

    def load_model(self, pred_path, target_path):
        # load prediction network and target network
        self.pred_net.load(pred_path)
        self.target_net.load(target_path)

    def save_buffer(self, buffer_path):
        self.replay_buffer.save_data(buffer_path)
        print("Successfully save buffer!")

    def load_buffer(self, buffer_path):
        # load data from the pkl file
        self.replay_buffer.read_list(buffer_path)

    def choose_action(self, s, epsilon, idling=None):
        # TODO: REPLACE THE FOLLOWING FAKE CODE WITH YOUR CODE
        x = torch.stack([torch.from_numpy(item[0]) for item in s]).float()
        state = torch.stack([torch.from_numpy(item[1]) for item in s]).float()
        # print(x.shape)
        # assert 1 == 2
        # print(x.shape)
        if USE_GPU:
            x = x.to(DEVICE)
            state = state.to(DEVICE)

        # # epsilon-greedy policy
        if input_actions > 0:
            if np.random.uniform() >= epsilon:
                # greedy case
                action_value_all = [
                    self.pred_net(
                        x,
                        state,
                        torch.full((x.size(0), 1), a_cur, device=DEVICE),
                    )
                    for a_cur in range(N_ACTIONS)
                ]
                action_value_all = torch.cat(action_value_all, dim=1)

                action = torch.argmax(action_value_all, dim=1).data.cpu().numpy()
                action = action.reshape(x.size(0))
            else:
                # random exploration case
                action = np.random.randint(0, N_ACTIONS, (x.size(0)))

        else:
            if np.random.uniform() >= epsilon:
                # greedy case
                action_value = self.pred_net(x, state)  # (N_ENVS, N_ACTIONS, N_QUANT)
                # print(action_value.shape)
                # assert 1 == 2
                action = torch.argmax(action_value, dim=1).data.cpu().numpy()

            else:
                # random exploration case
                action = np.random.randint(0, N_ACTIONS, (x.size(0)))
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
        self.learn_step_counter += 1

        # Target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, 1e-2)

        # Sample a batch
        (b_s_i, b_s_s, b_a, b_r, b_s_i_, b_s_s_, b_d) = self.replay_buffer.sample(
            BATCH_SIZE
        )
        # for x in (b_s_i, b_s_s, b_a, b_r, b_s_i_, b_s_s_, b_d):
        #     print(x.shape)
        # Convert to PyTorch tensors and possibly move to GPU
        b_s_i = torch.FloatTensor(b_s_i).to(DEVICE if USE_GPU else "cpu")
        b_s_s = torch.FloatTensor(b_s_s).to(DEVICE if USE_GPU else "cpu")
        b_a = torch.LongTensor(b_a).to(DEVICE if USE_GPU else "cpu")
        b_r = torch.FloatTensor(b_r).to(DEVICE if USE_GPU else "cpu")
        b_s_i_ = torch.FloatTensor(b_s_i_).to(DEVICE if USE_GPU else "cpu")
        b_s_s_ = torch.FloatTensor(b_s_s_).to(DEVICE if USE_GPU else "cpu")
        b_d = torch.FloatTensor(b_d).to(DEVICE if USE_GPU else "cpu")

        # Pass both sets of observations to the network and compute Q values
        # This assumes your network's forward method can accept and process both sets of observations
        # q_eval = self.pred_net(b_s_i, b_s_s).gather(1, b_a.unsqueeze(1))
        # q_next = self.target_net(b_s_i_, b_s_s_).max(1)[0].detach()
        # q_target = b_r + GAMMA * (1.0 - b_d) * q_next
        if input_actions > 0:
            q_eval = self.pred_net(b_s_i, b_s_s, b_a.reshape(-1, 1)).reshape(-1)
            mb_size = q_eval.size(0)
            # q_eval = torch.stack([q_eval[i][b_a[i]] for i in range(mb_size)])

            # optimal action value for current state
            q_next_all = [
                self.target_net(
                    b_s_i_,
                    b_s_s_,
                    torch.full((mb_size, 1), a_cur, device=DEVICE),
                )
                for a_cur in range(N_ACTIONS)
            ]
            q_next = torch.torch.cat(q_next_all, dim=1)
            # best_actions = q_next.argmax(dim=1)
            # q_next = torch.stack([q_next[i][best_actions[i]] for i in range(mb_size)])
            q_next = torch.max(q_next, -1)[0]
            # q_next = q_next.reshape(mb_size, 1)
        else:
            q_eval = self.pred_net(b_s_i, b_s_s)
            mb_size = q_eval.size(0)
            q_eval = torch.stack([q_eval[i][b_a[i]] for i in range(mb_size)])

            # optimal action value for current state
            q_next = self.target_net(b_s_i_, b_s_s_)
            # best_actions = q_next.argmax(dim=1)
            # q_next = torch.stack([q_next[i][best_actions[i]] for i in range(mb_size)])
            q_next = torch.max(q_next, -1)[0]

        q_target = b_r + GAMMA * (1.0 - b_d) * q_next
        q_target = q_target.detach()
        # Compute loss
        # print(q_eval.shape, q_target.shape, q_next.shape, b_d.shape)
        # assert 1 == 2
        loss = self.loss_function(q_eval, q_target)

        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
