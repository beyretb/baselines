from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException

from baselines.herSmallMovesMulti.util import convert_episode_to_batch_major, store_args


class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        subgoals = np.zeros((self.n_subgoals * self.n_steps_per_subgoal, self.rollout_batch_size, self.dims['g']))
        subgoals[(self.n_subgoals - 1) * self.n_steps_per_subgoal:, :, :] = self.g.copy()
        for n in range(self.n_subgoals-1,0,-1):
            sg = self.policy.get_subgoal(o, ag, subgoals[(n+1)*self.n_steps_per_subgoal-1],
                               goals_noise_eps=self.goals_noise_eps if not self.exploit else 0.,
                               goals_random_eps=self.goals_random_eps if not self.exploit else 0.)
            subgoals[(n-1)*self.n_steps_per_subgoal:n*self.n_steps_per_subgoal,:,:] = sg.copy()

        obs, achieved_goals, acts, goals, successes_g, successes_sg, rewards = [], [], [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in
                       self.info_keys]
        Qs = []

        for n in range(self.n_subgoals):

            success_sg = np.zeros(self.rollout_batch_size)
            for t_sub in range(self.n_steps_per_subgoal):

                t = n*self.n_steps_per_subgoal+t_sub
                sg = subgoals[t]

                policy_output = self.policy.get_actions(
                    o, ag, sg,
                    compute_Q=self.compute_Q,
                    noise_eps=self.noise_eps if not self.exploit else 0.,
                    random_eps=self.random_eps if not self.exploit else 0.,
                    use_target_net=self.use_target_net)

                if self.compute_Q:
                    u, Q = policy_output
                    Qs.append(Q)
                else:
                    u = policy_output

                if u.ndim == 1:
                    # The non-batched case should still have a reasonable shape.
                    u = u.reshape(1, -1)

                o_new = np.empty((self.rollout_batch_size, self.dims['o']))
                ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
                success_g = np.zeros(self.rollout_batch_size)
                reward = np.zeros(self.rollout_batch_size)
                # compute new states and observations
                for i in range(self.rollout_batch_size):
                    try:
                        # We fully ignore the reward here because it will have to be re-computed
                        # for HER.
                        curr_o_new, _, _, info = self.envs[i].step(u[i])
                        if 'is_success' in info:
                            success_g[i] = info['is_success']
                        o_new[i] = curr_o_new['observation']
                        ag_new[i] = curr_o_new['achieved_goal']
                        reward[i] = self.envs[i].compute_reward(achieved_goal=ag_new[i], desired_goal=sg[i], info=info)
                        success_sg[i]+= reward[i]==0
                        for idx, key in enumerate(self.info_keys):
                            info_values[idx][t, i] = info[key]
                        if self.render:
                            self.envs[i].env.sg = subgoals[t][i]
                            self.envs[i].render()
                    except MujocoException as e:
                        return self.generate_rollouts()

                if np.isnan(o_new).any():
                    self.logger.warning('NaN caught during rollout generation. Trying again...')
                    self.reset_all_rollouts()
                    return self.generate_rollouts()

                obs.append(o.copy())
                achieved_goals.append(ag.copy())
                successes_g.append(success_g.copy())
                acts.append(u.copy())
                goals.append(self.g.copy())
                rewards.append(reward.copy())
                o[...] = o_new
                ag[...] = ag_new

            successes_sg.append((success_sg>0).copy())

            if self.sg_regenerate and n < self.n_subgoals - 1:
                sg = self.policy.get_subgoal(o, ag,
                                             subgoals[(n + 1) * self.n_steps_per_subgoal - 1],
                                             goals_noise_eps=self.goals_noise_eps if not self.exploit else 0.,
                                             goals_random_eps=self.goals_random_eps if not self.exploit else 0.)
                subgoals[(n - 1) * self.n_steps_per_subgoal:n * self.n_steps_per_subgoal, :, :] = sg.copy()

        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals,
                       sg=subgoals,
                       r=rewards,
                       sg_success=successes_sg)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        # successful = np.array(successes)[-1, :]
        # assert successful.shape == (self.rollout_batch_size,)
        # success_rate = np.mean(successful)
        # self.success_history.append(success_rate)
        self.success_history.append(np.mean(successes_sg))
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)
