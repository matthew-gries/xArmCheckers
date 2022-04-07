from typing import Tuple, Dict, Optional, Callable
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import gym
import gym.spaces
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from checkers.game import Game

from xarm_checkers.checkers.utils import game_as_numpy, render_game

class CheckersEnv(gym.Env):
    def __init__(self,
                 episode_length: int=50,
                ) -> None:
        '''Gym checkers environment

        Parameters
        ----------
        episode_length
            number of actions that agent can take before environment is reset
        '''

        self.checkers_game = Game()
        self.episode_length = episode_length

        # 32 element observation space, each element represents a spot on the checkers board a piece can be at (0 - 31)
        # Values correspond to the following:
        # * -2 -> P2 king
        # * -1 -> P2 normal
        # * 0  -> Empty
        # * 1  -> P1 normal
        # * 2  -> P1 king
        self.observation_space = gym.spaces.Box(low=-2, high=2, shape=(32,), dtype=np.int8)

        # 4 x 32 actions space
        # - First element is the space to move to (encoded as 0 - 31) if we are moving down and to the left
        # - Second element is the space to move to (encoded as 0 - 31) if we are moving down and to the right
        # - Third element is the space to move to (encoded as 0 - 31) if we are moving up and to the left
        # - Fourth element is the space to move to (encoded as 0 - 31) if we are moving up and to the right
        self.action_space = gym.spaces.Tuple([
            gym.spaces.Discrete(32),
            gym.spaces.Discrete(32),
            gym.spaces.Discrete(32),
            gym.spaces.Discrete(32)
        ])

        self._move_map = {i: {j: [] for j in range(4)} for i in range(1, 33)}

        self._fill_move_map()

    def _fill_move_map(self):

        for i in range(1, 33):
            for j in range(4):
                # fill lower left
                if j == 0:
                    # nothing lower than these spots
                    if i in range(29, 33):
                        continue
                    # nothing further left
                    elif i in [5, 13, 21]:
                        continue
                    # just one piece
                    elif i in [1, 9, 17, 25, 26, 27, 28]:
                        self._move_map[i][j] = [i+4]
                    # all other pieces
                    elif i in [6, 7, 8, 14, 15, 16, 22, 23, 24]:
                        self._move_map[i][j] = [i+3, i+7]
                    else:
                        self._move_map[i][j] = [i+4, i+7]
                # fill lower right
                elif j == 1:
                    # nothing lower than these spots
                    if i in range(29, 33):
                        continue
                    # nothing further right
                    elif i in [4, 12, 20]:
                        continue
                    # just one piece
                    elif i in [8, 16, 24]:
                        self._move_map[i][j] = [i+4]
                    elif i in [25, 26, 27]:
                        self._move_map[i][j] = [i+5]
                    # all other pieces
                    elif i in [1, 2, 3, 9, 10, 11, 17, 18, 19]:
                        self._move_map[i][j] = [i+5, i+9]
                    else:
                        self._move_map[i][j] = [i+4, i+9]
                # fill upper left
                if j == 2:
                    # nothing higher than these spots
                    if i in range(1, 5):
                        continue
                    # nothing further left
                    elif i in [5, 13, 21, 29]:
                        continue
                    # just one piece
                    elif i in [6, 7, 8]:
                        self._move_map[i][j] = [i-5]
                    elif i in [9, 17, 25]:
                        self._move_map[i][j] = [i-4]
                    # all other pieces
                    elif i in [10, 11, 12, 18, 19, 20, 26, 27, 28]:
                        self._move_map[i][j] = [i-4, i-9]
                    else:
                        self._move_map[i][j] = [i-5, i-9]
                # fill upper right
                if j == 3:
                    # nothing higher than these spots
                    if i in range(1, 5):
                        continue
                    # nothing further right
                    elif i in [12, 20, 28]:
                        continue
                    # just one piece
                    elif i in [5, 6, 7, 8, 16, 24, 32]:
                        self._move_map[i][j] = [i-4]
                    # all other pieces
                    elif i in [9, 10, 11, 17, 18, 19, 25, 26, 27]:
                        self._move_map[i][j] = [i-3, i-7]
                    else:
                        self._move_map[i][j] = [i-4, i-7]

    def reset(self) -> np.ndarray:
        '''Sets the game state to the initial state of checkers (TODO should we always do this)
        and sets time step counter to 0

        Returns
        -------
        observation of the checkers game
        '''
        self.checkers_game = Game()
        self.t_step = 0

        return self.get_obs()

    @staticmethod
    def _get_max_subsequent_jumps_count(game: Game, current_player: int, move: Tuple[int, int]) -> Tuple[int, Game]:
        game.move(move)
        if game.whose_turn() != current_player:
            return 1, game
        else:
            moves = game.get_possible_moves()
            res = [CheckersEnv._get_max_subsequent_jumps_count(copy.deepcopy(game), current_player, m) for m in moves]
            counts = [r[0] for r in res]
            idx = np.argmax(counts)
            return 1 + res[idx][0], res[idx][1]

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        '''
        Performs an action by moving from a position to another position (represented as a
        tuple where the first element is the 0-31 position to move from and the second position is
        the 0-31 position to move to)

        If the action results in a jump, we take the sequence of jumps that results in the longest jump.

        As such, the returned observation will always have a different player than the previous observation.

        Returns
        -------
        tuple of (obs, reward, done, info)
        '''
        # assert self.action_space.contains(action)
        self.t_step += 1

        prev_player = self.checkers_game.whose_turn()

        # Make the move
        self.checkers_game.move(action)

        # If we still have more jumps, recursively find the longest jump sequence
        if self.checkers_game.whose_turn() == prev_player:
            game = self.checkers_game
            moves = game.get_possible_moves()
            res = [CheckersEnv._get_max_subsequent_jumps_count(copy.deepcopy(game), prev_player, m) for m in moves]
            counts = [r[0] for r in res]
            idx = np.argmax(counts)
            self.checkers_game = res[idx][1]
            # TODO do we update t_step?


        obs = self.get_obs()
        reward = self.get_reward()
        done = self.is_done()
        info = {}

        return obs, reward, done, info

    def get_obs(self) -> np.ndarray:
        '''Returns observation which is a numpy representation of the game
        '''
        return game_as_numpy(self.checkers_game)

    def get_reward(self) -> float:
        '''Calculates reward based on if the current state has a winner, and which player is
        the winner
        '''
        if self.checkers_game.get_winner() == 1:
            return 1.0
        elif self.checkers_game.get_winner() == 2:
            return -1.0
        else:
            return 0.0

    def is_done(self) -> bool:
        '''Environment should be reset if episode is over or we have reached a terminal state
        '''
        return self.t_step >= self.episode_length \
                or self.checkers_game.is_over()

    def get_action_from_action_space_idx(self, from_pos: int, idx: int) -> Tuple[int, int]:
        """
        Get the action from the given position in the current state to the position that is in the
        direction specified by the index (see the action space for what this means)

        If a move cannot be made, returns None
        """

        potential_positions = self._move_map[from_pos][idx]
        current_legal_moves = self.checkers_game.get_possible_moves()

        # potential positions considers scenarios for moves and jumps, will be either one of the other
        # in the legal moveset
        for to_pos in potential_positions:
            if (from_pos, to_pos) in current_legal_moves:
                return (from_pos, to_pos)

        return None

def watch_policy(env: CheckersEnv, policy: Optional[Callable]=None):
    '''Rolls out policy in environment and renders in GUI.  If policy is not
    provided then random policy is used

    Note
    ----
    environment must be initialized with `render=True`
    '''
    if policy is None:
        policy = lambda s: env.action_space.sample()

    s = env.reset()
    while 1:
        a = policy(s)
        sp, r, done, _ = env.step(a)
        env.render(0.1)

        s = sp
        if done:
            time.sleep(2.)
            s = env.reset()


class QNetwork(nn.Module):
    def __init__(self, env: CheckersEnv) -> None:
        '''Q-Network instantiated as 3-layer MLP with 64 units
        '''
        super().__init__()

        self.state_vector_length = 32
        self.action_vector_length = 4 * 32

        self.layers = nn.Sequential(
            nn.Linear(self.state_vector_length, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, self.action_vector_length)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, s: Tensor) -> Tensor:
        '''Perform forward pass

        Parameters
        ----------
        s
            state tensor; shape=(B,|S|); dtype=float32
        Returns
        -------
        tensor of q values for each action; shape=(B,|A|); dtype=float32
        '''
        return self.layers(s)

    @torch.no_grad()
    def predict(self, s: Tensor) -> Tensor:
        '''Get the q_values from the network, given the state
        '''
        q_vals = self.forward(s)
        return q_vals    

    def compute_loss(self, q_pred: Tensor, q_target: Tensor) -> Tensor:
        return self.loss_fn(q_pred, q_target)


class ReplayBuffer:
    def __init__(self, size: int, state_dim: int) -> None:
        self.data = {'s' : np.zeros((size, state_dim), dtype=np.float32),
                     'a' : np.zeros((size), dtype=np.int32),
                     'r' : np.zeros((size), dtype=np.float32),
                     'sp' : np.zeros((size, state_dim), dtype=np.float32),
                     'd' : np.zeros((size), dtype=np.bool8),
                    }

        self.size = size
        self.length = 0
        self._idx = 0

    def add_transition(self, s: np.ndarray, a: int, r: float,
                       sp: np.ndarray, d: bool) -> None:
        self.data['s'][self._idx] = s
        self.data['a'][self._idx] = a
        self.data['r'][self._idx] = r
        self.data['sp'][self._idx] = sp
        self.data['d'][self._idx] = d

        self._idx = (self._idx + 1) % self.size
        self.length = min(self.length + 1, self.size)

    def sample(self, batch_size: int) -> Tuple:
        idxs = np.random.randint(0, self.length, batch_size)

        s = self.data['s'][idxs]
        a = self.data['a'][idxs]
        r = self.data['r'][idxs]
        sp = self.data['sp'][idxs]
        d = self.data['d'][idxs]

        return s, a, r, sp, d


class Agent:
    def __init__(self,
                 env: CheckersEnv,
                 gamma: float=1.,
                 learning_rate: float=5e-4,
                 buffer_size: int=50000,
                 batch_size: int=128,
                 initial_epsilon: float=1.,
                 final_epsilon: float=0.01,
                 exploration_fraction: float=0.9,
                 target_network_update_freq: int=1000,
                 seed: int=0,
                 device: str='cpu',
                ) -> None:
        '''Agent that learns policy using DQN algorithm
        '''
        self.env = env

        assert 0 < gamma <= 1., 'Discount factor (gamma) must be in range (0,1]'
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.exploration_fraction = exploration_fraction

        self.buffer = ReplayBuffer(buffer_size, self.env.observation_space.shape[0])

        self.device = device
        self.network = QNetwork().to(device)
        self.target_network = QNetwork().to(device)
        self.hard_target_update()

        self.optim = torch.optim.Adam(self.network.parameters(),
                                      lr= learning_rate)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    def train(self, num_steps: int) -> None:
        '''Trains q-network for given number of environment steps, plots
        rewards and loss curve
        '''
        rewards_data = []
        success_data = []
        loss_data = []

        episode_count = 0
        episode_rewards = 0
        opt_count = 0
        s = self.env.reset()

        pbar = tqdm(range(1, num_steps+1))
        for step in pbar:
            epsilon = self.compute_epsilon(step/(self.exploration_fraction*num_steps))
            a = self.select_action(s, epsilon)

            sp, r, done, info = self.env.step(a)
            episode_rewards += r

            self.buffer.add_transition(s=s, a=a, r=r, sp=sp, d=done)

            # optimize
            if self.buffer.length > self.batch_size:
                loss = self.optimize()
                opt_count += 1
                loss_data.append(loss)
                if opt_count % self.target_network_update_freq == 0:
                    self.hard_target_update()

            s = sp.copy()
            if done:
                s = self.env.reset()
                rewards_data.append(episode_rewards)
                success_data.append(info['success'])
                episode_rewards = 0
                episode_count += 1
                avg_success = np.mean(success_data[-min(episode_count, 50):])
                pbar.set_description(f'Success = {avg_success:.1%}')

        f, axs = plt.subplots(1,3, figsize=(7.5,2))
        axs[0].plot(np.convolve(rewards_data, np.ones(50)/50, 'valid'))
        axs[0].set_xlabel('episodes')
        axs[0].set_ylabel('sum of rewards')
        axs[1].plot(np.convolve(success_data, np.ones(50)/50, 'valid'))
        axs[1].set_xlabel('episodes')
        axs[1].set_ylabel('success rate')
        axs[2].plot(np.convolve(loss_data, np.ones(200)/200, 'valid'))
        axs[2].set_xlabel('opt steps')
        axs[2].set_ylabel('td loss')
        plt.tight_layout()
        plt.show()

    def optimize(self) -> float:
        '''Optimize Q-network by minimizing td-error on mini-batch sampled
        from replay buffer
        '''
        s,a,r,sp,d = self.buffer.sample(self.batch_size)

        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        sp = torch.tensor(sp, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        q_pred = self.network(s).gather(1, a.unsqueeze(1)).squeeze()

        with torch.no_grad():
            q_target = r + self.gamma * torch.max(self.target_network(sp), dim=1)[0]

        self.optim.zero_grad()

        assert q_pred.shape == q_target.shape
        loss = self.network.compute_loss(q_pred, q_target)
        loss.backward()

        # it is common to clip gradient to prevent instability
        nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optim.step()
        return loss.item()

    def select_action(self, state: np.ndarray, epsilon: float=0.) -> Tuple[int, int]:
        '''Performs e-greedy action selection, TODO what type of action selection'''
        # TODO random sampling -> do the same type of logic in policy
        return self.policy(state)

    def compute_epsilon(self, fraction: float) -> float:
        '''Compute epsilon value based on fraction of training steps'''
        fraction = np.clip(fraction, 0., 1.)
        return (1-fraction) * self.initial_epsilon + fraction * self.final_epsilon

    def hard_target_update(self):
        '''Copy weights of q-network to target q-network'''
        self.target_network.load_state_dict(self.network.state_dict())

    def policy(self, state: np.ndarray) -> Tuple[int, int]:
        '''Get the next action to take given the '''
        t_state = torch.tensor(state, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        q_values = self.network.predict(t_state, self.env)

        q_value_idx = torch.argmax(q_values, dim=1)

        from_pos = q_value_idx // 4
        direction = q_value_idx % 4

        action = self.env.get_action_from_action_space_idx(from_pos, direction)

        # if the action is not legal, pick a random action
        if action is None:
            action = self.env.get_random_action()

        return action

if __name__ == "__main__":
    env = CheckersEnv()

    agent = Agent(env,
                  gamma=0.98,
                  learning_rate=1e-3,
                  buffer_size=10000,
                  initial_epsilon=0.1,
                  final_epsilon=0.01,
                  exploration_fraction=0.9,
                  target_network_update_freq=1000,
                  batch_size=256,
                  device='cpu',
                 )
    agent.train(20000)

    watch_policy(env, agent.policy)

