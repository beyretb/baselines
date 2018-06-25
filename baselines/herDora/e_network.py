import tensorflow as tf
from baselines.herDora.util import store_args


class ENetwork:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):

        """ The E Network for continuous case, adapted from https://arxiv.org/abs/1804.04012

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)

        with tf.variable_scope('E'):
            # for critic training
            input = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            for i in range(self.layers - 1):
                input = tf.layers.dense(inputs=input,
                                        units=self.hidden,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        name='_' + str(i))
                input = tf.nn.relu(input)
            input = tf.layers.dense(inputs=input,
                                    units=1,
                                    kernel_initializer=tf.zeros_initializer(),
                                    name='_' + str(self.layers))
            self.E_tf = tf.nn.tanh(input)