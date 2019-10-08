"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.99,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=0.01,
            output_graph=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.name="meta_"
        self.size=[self.n_features,10,10,10,self.n_actions]

        self.w_initializer, self.b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
    def add_layer(self,input,input_length,output_length,c_name,name,layer_ID,activation_fun=None):
        print(layer_ID,input_length,output_length)
        with tf.variable_scope(layer_ID):
            w=tf.get_variable(self.name+name+"_w_"+layer_ID,[input_length,output_length],initializer=self.w_initializer,collections=c_name)
            b=tf.get_variable(self.name+name+"_b_"+layer_ID,[1,output_length],initializer=self.b_initializer,collections=c_name)
            if activation_fun==None:
                output = tf.matmul(input,w)+b
            else:
                output = activation_fun(tf.matmul(input,w)+b)
        return output
    def add_net(self,input,c_name,name):
        layer=input
        with tf.variable_scope(name):
            for i in range(len(self.size)-2):
                layer=self.add_layer(layer,self.size[i],self.size[i+1],c_name,name,"layer_"+str(i+1),activation_fun=tf.nn.relu)
            with tf.variable_scope(self.name + "_" + name + '_Value'):
                w3 = tf.get_variable(self.name + "_" + name + "w_" + str(len(self.size)), [self.size[-2], 1],
                                     initializer=self.w_initializer, collections=c_name)
                b3 = tf.get_variable(self.name + "_" + name + "b_" + str(len(self.size)), [1, 1], initializer=self.b_initializer,
                                     collections=c_name)
                V = tf.matmul(layer, w3) + b3

            with tf.variable_scope(self.name + "_" + name + '_Advantage'):
                w3 = tf.get_variable(self.name + "_" + name + "w_" + str(len(self.size)), [self.size[-2], self.size[-1]],
                                     initializer=self.w_initializer, collections=c_name)
                b3 = tf.get_variable(self.name + "_" + name + "b_" + str(len(self.size)), [1, self.size[-1]],
                                     initializer=self.b_initializer, collections=c_name)
                A = tf.matmul(layer, w3) + b3

            with tf.variable_scope('dueling_Q'):
                output = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)
            #output=self.add_layer(layer,self.size[-2],self.size[-1],c_name,name,"layer_"+str(len(self.size)-1))
        return output
    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        c_names=['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

        self.q_eval=self.add_net(self.s,c_name=c_names,name="eval_net")
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        self.q_next=self.add_net(self.s_,c_names,"next_net")
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],  # next observation
                       self.s: batch_memory[:, -self.n_features:]})  # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        max_act4next = np.argmax(q_eval4next,
                                 axis=1)  # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return self.cost

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
if __name__=="__main__":
    rl=DeepQNetwork(n_actions=3,n_features=8)



