from collections import deque
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import gym

from network import create_network
from train_ops import *
from utils import *

from IPython.core.debugger import Pdb

G = 0.99
N_ACTIONS = 4
ACTIONS = np.arange(N_ACTIONS)

def list_set(l, i, val):
    assert(len(l) == i)
    l.append(val)

class Worker:

    def __init__(self, sess, worker_n, env_name, summary_writer):
        self.sess = sess
        self.env = gym.make(env_name)

        worker_scope = "worker_%d" % worker_n
        self.network = create_network(worker_scope)
        self.summary_writer = summary_writer
        self.scope = worker_scope

        self.reward = tf.Variable(0.0)
        self.reward_summary = tf.summary.scalar('reward', self.reward)

        policy_optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)

        self.update_policy_gradients, self.apply_policy_gradients, self.zero_policy_gradients, self.grad_bufs_policy = \
            create_train_ops(self.network.policy_loss,
                             policy_optimizer,
                             update_scope=worker_scope,
                             apply_scope='global')

        self.update_value_gradients, self.apply_value_gradients, self.zero_value_gradients, self.grad_bufs_value = \
            create_train_ops(self.network.value_loss,
                             value_optimizer,
                             update_scope=worker_scope,
                             apply_scope='global')

        self.init_copy_ops()

        self.reset_env()

        self.t_max = 10000
        self.steps = 0
        self.episode_rewards = []
        self.render = False

        self.value_log = deque(maxlen=100)
        self.fig = None

    def reset_env(self):
        self.o = self.env.reset()

    def log_rewards(self):
        reward_sum = sum(self.episode_rewards)
        print("Reward sum was", reward_sum)
        self.sess.run(tf.assign(self.reward, reward_sum))
        summ = self.sess.run(self.reward_summary)
        self.summary_writer.add_summary(summ, self.steps)


    def init_copy_ops(self):
        from_tvs = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='global')
        to_tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=self.scope)

        from_dict = {var.name: var for var in from_tvs}
        to_dict = {var.name: var for var in to_tvs}
        copy_ops = []
        for to_name, to_var in to_dict.items():
            from_name = to_name.replace(self.scope, 'global')
            from_var = from_dict[from_name]
            op = to_var.assign(from_var.value())
            copy_ops.append(op)

        self.copy_ops = copy_ops


    def sync_network(self):
        self.sess.run(self.copy_ops)


    def value_graph(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.fig.set_size_inches(2, 2)
            self.ax.set_xlim([0, 100])
            self.ax.set_ylim([0, 2.0])
            self.line, = self.ax.plot([], [])

            self.fig.show()
            self.fig.canvas.draw()
            self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        self.fig.canvas.restore_region(self.bg)

        ydata = list(self.value_log)
        xdata = list(range(len(self.value_log)))
        self.line.set_data(xdata, ydata)

        self.ax.draw_artist(self.line)
        self.fig.canvas.update()
        self.fig.canvas.flush_events()

    def run_step(self):
        states = []
        actions = []
        rewards = []
        i = 0

        self.sess.run([self.zero_policy_gradients,
                  self.zero_value_gradients])
        self.sync_network()

        list_set(states, i, self.o)

        done = False
        while not done and i < self.t_max:
            feed_dict = {self.network.s: [[self.o]]}
            a_p = self.sess.run(self.network.a_softmax, feed_dict=feed_dict)[0]
            a = np.random.choice(ACTIONS, p=a_p)
            list_set(actions, i, a)

            o, r, done, _ = self.env.step(a)

            if self.render:
                self.env.render()
                feed_dict = {self.network.s: [[o]]}
                v = self.sess.run(self.network.graph_v, feed_dict=feed_dict)[0]
                self.value_log.append(v)
                self.value_graph()

            if r != 0:
                print("Got reward", r)
            self.o = o
            self.episode_rewards.append(r)
            list_set(rewards, i, r)
            list_set(states, i + 1, self.o)

            i += 1

        if done:
            print("Episode done")
            self.log_rewards()
            self.episode_rewards = []

        # Calculate initial value for R
        if done:
            # Terminal state
            r = 0
        else:
            # Non-terminal state
            # Estimate the value of the current state using the value network
            # (states[i]: the last state)
            s = np.moveaxis(states[i], source=0, destination=-1)
            feed_dict = {self.network.s: [s]}
            r = self.sess.run(self.network.graph_v, feed_dict=feed_dict)[0]

        # i - 1 to 0
        # (Why start from i - 1, rather than i?
        #  So that we miss out the last state.)
        for j in reversed(range(i)):
            s = states[j]
            r = rewards[j] + G * r
            feed_dict = {self.network.s: [[s]],
                         self.network.a: [actions[j]],
                         self.network.r: [r]}

            self.sess.run([self.update_policy_gradients,
                      self.update_value_gradients],
                      feed_dict)

        # Perform update of global parameters
        self.sess.run([self.apply_value_gradients])
        self.sess.run([self.zero_policy_gradients,
                       self.zero_value_gradients])

        self.steps += 1

        return done
