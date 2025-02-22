from env import Env
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F



# Define model
class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.loss = nn.MSELoss()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


# Deep Q-Learning
class CarDQL():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.005  # learning rate (alpha)
    discount_factor_g = 0.95  # discount rate (gamma)
    network_sync_rate = 50000  # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 50000  # size of replay memory
    mini_batch_size = 64  # size of the training data set sampled from the replay memory


    # Neural Network
    loss_fn = nn.MSELoss()  # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None  # NN Optimizer. Initialize later.

    # Train the environment
    def train(self, episodes, steps):
        # Create instance
        env = Env()
        num_states = 6 # expecting 2: position & velocity
        num_actions = 3

        epsilon = 1  # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DeepQNetwork(input_dims=num_states, fc1_dims=64,fc2_dims=64, n_actions=num_actions)
        target_dqn = DeepQNetwork(input_dims=num_states, fc1_dims=64,fc2_dims=64, n_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = []

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.

        for i in range(episodes):
            state = env.reset()  # Initialize to state 0
            done = False
            pre_action = None
            rewards = 0
            step_count = 0
            best_rewards = -10
            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            for t in range(steps):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = np.random.choice([0,1,2])  # actions: 0=left,1=left,2=right
                else:
                    # select best action
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                # Execute action
                reward = env.setReward(state,pre_action,action)
                new_state, done = env.step(action)

                # Accumulate reward
                rewards += reward
                pre_action = action
                # Save experience into memory
                memory.append((state, action, new_state, reward, done))

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count += 1
                if t%20 ==0:
                    print("episodes: %d; score %d; step: %d", i, rewards, t)
                if t == 1000 or rewards <= -30 or rewards >= 30:
                    done = True
                    print("episodes: %d done; total_score %d; total_step: %d", i, rewards, t)
                    break

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(rewards)

            # Graph training progress
            if (i != 0 and i % 1000 == 0):
                print(f'Episode {i} Epsilon {epsilon}')

                self.plot_progress(rewards_per_episode, epsilon_history)

            if rewards > best_rewards:
                best_rewards = rewards
                print(f'Best rewards so far: {best_rewards}')
                # Save policy
                torch.save(policy_dqn.state_dict(), f"car_dql_{i}.pt")

            # Check if enough experience has been collected
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1 / episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

                    # Close environment
        quit()

    def plot_progress(self, rewards_per_episode, epsilon_history):
        # Create new graph
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        # rewards_curve = np.zeros(len(rewards_per_episode))
        # for x in range(len(rewards_per_episode)):
        # rewards_curve[x] = np.min(rewards_per_episode[max(0, x-10):(x+1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        # plt.plot(sum_rewards)
        plt.plot(rewards_per_episode)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)

        # Save plots
        plt.savefig('car_dql.png')

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                # Agent receive reward of 0 for reaching goal.
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state))
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state) -> torch.Tensor:
        dis_1 = state[0]
        dis_2 = state[1]
        dis_3 = state[2]
        dis_4 = state[3]
        dis_5 = state[4]
        state_distacne = state[5]
        state_yaw = state[6]
        return torch.FloatTensor([dis_1, dis_2,dis_3,dis_4,dis_5, state_distacne, state_yaw])
