import threading

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, sample_goal_transitions,
                 n_subgoals, sample_method, reward_type):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
            n_subgoals: number of subgoals
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions
        self.sample_goal_transitions = sample_goal_transitions
        self.n_subgoals = n_subgoals
        self.n_steps_per_subgoal = int(T/n_subgoals)
        self.sample_method = sample_method
        self.reward_type = reward_type

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}
        # self.buffers_G = {key: np.empty([self.size, n_subgoals, shape[1]])
        #                 for key, shape in buffer_shapes.items()}
        self.buffers_G = {'ag': np.empty([self.size, n_subgoals+1, buffer_shapes['g'][1]]),
                          'g':np.empty([self.size, n_subgoals, buffer_shapes['g'][1]]),
                          'o': np.empty([self.size, n_subgoals, buffer_shapes['o'][1]]),
                          'r': np.empty([self.size, n_subgoals]),
                          'sg': np.empty([self.size, n_subgoals, buffer_shapes['g'][1]]),
                          'sg_success':np.empty([self.size, n_subgoals])}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        # self.lock = threading.Lock()

    @property
    def full(self):
        # with self.lock:
        return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        # with self.lock:
        assert self.current_size > 0
        for key in self.buffers.keys():
            buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions = self.sample_transitions(buffers, batch_size, self.n_subgoals)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def sample_goal(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        # with self.lock:
        assert self.current_size > 0
        for key in self.buffers_G.keys():
            buffers[key] = self.buffers_G[key][:self.current_size]

        buffers['o_2'] = buffers['o'][1:, :, :]
        buffers['sg_2'] = buffers['sg'][1:, :, :]
        # buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions = self.sample_goal_transitions(buffers, batch_size, self.sample_method, self.reward_type)

        # for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
        #     assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        # with self.lock:
        idxs = self._get_storage_idx(batch_size)

        rewards = episode_batch.pop('r')
        sg_success = episode_batch.pop('sg_success')

        # load inputs into buffers
        for key in self.buffers.keys():
            self.buffers[key][idxs] = episode_batch[key]

        self.buffers_G['g'][idxs] = episode_batch['g'][:,np.array(range(self.n_subgoals))*self.n_steps_per_subgoal]
        self.buffers_G['o'][idxs] = episode_batch['o'][:, np.array(range(self.n_subgoals)) * self.n_steps_per_subgoal]
        self.buffers_G['sg'][idxs] = episode_batch['sg'][:, np.array(range(self.n_subgoals)) * self.n_steps_per_subgoal]
        self.buffers_G['ag'][idxs] = episode_batch['ag'][:, np.array(range(self.n_subgoals+1)) * self.n_steps_per_subgoal]
        for i in range(self.n_subgoals):
            self.buffers_G['r'][idxs,i] = np.sum(rewards[:,i*self.n_steps_per_subgoal:(i+1)*self.n_steps_per_subgoal], axis=1).copy()
        self.buffers_G['sg_success'][idxs] = sg_success

        self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx
