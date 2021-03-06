from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.herSmallMovesMulti.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from baselines.herSmallMovesMulti.normalizer import Normalizer
from baselines.herSmallMovesMulti.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, reuse=False, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.herSmallMovesMulti.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)
        self.create_actor_critic_goals = import_function(self.goals_network_class)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model.
        # First the lower level actor one
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            if key!='g':
                stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'sg']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes
        # stage placeholder
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

        # Then the higher level one
        stage_shapes_G = OrderedDict()
        for key in sorted(['o', 'g', 'sg']):
            stage_shapes_G[key] = (None, *input_shapes[key])
        for key in ['o', 'sg']:
            stage_shapes_G[key + '_2'] = stage_shapes_G[key]
        stage_shapes_G['r'] = (None,)
        self.stage_shapes_G = stage_shapes_G
        # stage placeholder
        with tf.variable_scope(self.scope):
            self.staging_G_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes_G.keys()],
                shapes=list(self.stage_shapes_G.values()))
            self.buffer_G_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes_G.values()]
            self.stage_G_op = self.staging_G_tf.put(self.buffer_G_ph_tf)

            self._create_network(reuse=reuse)

        self.goal_noise = self._GaussianNoise

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T+1, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions,
                                   self.sample_goal_transitions, self.n_subgoals, self.sample_method, self.reward_type)

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def _preprocess_sg(self, sg):
        return np.clip(sg, -self.clip_obs, self.clip_obs)

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, ag, g) # clip observations and goals
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        # ret = action given by the current policy (eval of NN)
        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        # Below: for each mini-batch we take action u (the one given by the policy) with probability
        # 1-random_eps, and a random action (u + random_action - u) with probability random_eps
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions, self.n_subgoals)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def get_subgoal(self, o, ag, g, goals_noise_eps = 0., goals_random_eps = 0.):

        if np.random.rand()<goals_random_eps:
            noise = self.goal_noise()
            sg = ag + noise
        else:
            o,g = self._preprocess_og(o,g,g)
            policy = self.target_G
            vals = [policy.pi_tf]
            feed = {
                policy.o_tf: o.reshape(-1, self.dimo),
                policy.g_tf: g.reshape(-1, self.dimg),
                policy.sg_tf: np.zeros((g.size // self.dimg, self.dimg), dtype=np.float32)
            }
            d = self.sess.run(vals,feed_dict=feed)
            sg = g + d
        sg = np.clip(sg, -self.clip_obs, self.clip_obs)
        return sg

    def _GaussianNoise(self):
        mean = 0.0
        sigma = 0.05
        return sigma*np.random.randn(self.dimg) + mean


    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _sync_G_optimizers(self):
        self.Q_G_adam.sync()
        self.pi_G_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _grads_G(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_G_tf,
            self.main_G.Q_pi_tf,
            self.Q_grad_G_tf,
            self.pi_grad_G_tf
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def _update_G(self, Q_grad, pi_grad):
        self.Q_G_adam.update(Q_grad, self.Q_lr)
        self.pi_G_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self):
        transitions = self.buffer.sample(self.batch_size)
        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        sg = transitions['sg']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)
        transitions['sg'] = self._preprocess_sg(sg)
        transitions['sg_2'] = self._preprocess_sg(sg)
        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        return critic_loss, actor_loss

    def train_goal(self, stage=True):
        if stage:
            self.stage_goal_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads_G()
        self._update_G(Q_grad, pi_grad)
        return critic_loss, actor_loss

    def sample_goal_batch(self):
        transitions = self.buffer.sample_goal(self.batch_size)
        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        transitions['o'], transitions['g'] = self._preprocess_og(o,g,g)
        transitions_batch = [transitions[key] for key in self.stage_shapes_G.keys()]
        return transitions_batch

    def stage_goal_batch(self, batch=None):
        if batch is None:
            batch = self.sample_goal_batch()
        assert len(self.buffer_G_ph_tf) == len(batch)
        self.sess.run(self.stage_G_op, feed_dict=dict(zip(self.buffer_G_ph_tf, batch)))

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)
        self.sess.run(self.init_target_G_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def update_target_net_G(self):
        self.sess.run(self.update_target_G_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # ============= Q Network ===========================================================================

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['sg'] = batch_tf['sg_2']
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))

        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)

        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))

        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]),
                zip(self.target_vars, self.main_vars)))

        # ============= g Network ===========================================================================

        # mini-batch sampling.
        batch_G = self.staging_G_tf.get()
        batch_G_tf = OrderedDict([(key, batch_G[i])
                                for i, key in enumerate(self.stage_shapes_G.keys())])
        batch_G_tf['r'] = tf.reshape(batch_G_tf['r'], [-1, 1])

        # networks
        with tf.variable_scope('main_G') as vs:
            if reuse:
                vs.reuse_variables()
            self.main_G = self.create_actor_critic_goals(batch_G_tf, net_type='main_G', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target_G') as vs:
            if reuse:
                vs.reuse_variables()
            target_G_batch_tf = batch_G_tf.copy()
            target_G_batch_tf['o'] = batch_G_tf['o_2']
            target_G_batch_tf['sg'] = batch_G_tf['sg_2']
            self.target_G = self.create_actor_critic_goals(target_G_batch_tf, net_type='target_G', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main_G")) == len(self._vars("target_G"))

        target_G_Q_pi_tf = self.target_G.Q_pi_tf
        target_G_tf = tf.clip_by_value(batch_G_tf['r'] + self.gamma * target_G_Q_pi_tf, *clip_range)
        self.Q_loss_G_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_G_tf) - self.main_G.Q_tf))
        self.pi_loss_G_tf = -tf.reduce_mean(self.main_G.Q_pi_tf)
        self.pi_loss_G_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main_G.pi_tf / self.max_d))

        Q_grads_G_tf = tf.gradients(self.Q_loss_G_tf, self._vars('main_G/Q'))
        pi_grads_G_tf = tf.gradients(self.pi_loss_G_tf, self._vars('main_G/pi'))
        assert len(self._vars('main_G/Q')) == len(Q_grads_G_tf)
        assert len(self._vars('main_G/pi')) == len(pi_grads_G_tf)

        self.Q_grads_vars_G_tf = zip(Q_grads_G_tf, self._vars('main_G/Q'))
        self.pi_grads_vars_G_tf = zip(pi_grads_G_tf, self._vars('main_G/pi'))

        self.Q_grad_G_tf = flatten_grads(grads=Q_grads_G_tf, var_list=self._vars('main_G/Q'))
        self.pi_grad_G_tf = flatten_grads(grads=pi_grads_G_tf, var_list=self._vars('main_G/pi'))
        self.Q_G_adam = MpiAdam(self._vars('main_G/Q'), scale_grad_by_procs=False)
        self.pi_G_adam = MpiAdam(self._vars('main_G/pi'), scale_grad_by_procs=False)

        self.main_G_vars = self._vars('main_G/Q') + self._vars('main_G/pi')
        self.target_G_vars = self._vars('target_G/Q') + self._vars('target_G/pi')
        self.stats_G_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_G_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_G_vars, self.main_G_vars)))
        self.update_target_G_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]),
                zip(self.target_G_vars, self.main_G_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._sync_G_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions', 'sample_goal_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None
        if 'sample_goal_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_goal_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)
