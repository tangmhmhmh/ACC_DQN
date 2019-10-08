#The DDQN ,is the sum of Dueling DQN and Double DQN
#Design: TangMingHong
#Data:   2019.10.05
#*********************************************************************************************************************
import tensorflow as tf
from DQN.DQN_Setting import DQN_Setting
import numpy as np
class Double_Dueling_DQN():
    def __init__(self):
        s = DQN_Setting()
        self.n_action=s.n_action
        self.n_feature=s.n_feature
        self.lr=s.learning_rate
        self.e_greedy=s.e_greedy
        self.e_greedy_increment=s.e_greedy_increment
        self.gamma=s.gamma
        self.name="meta"
        self.save_graph=s.output_graph
        self.graph_path=s.graph_path
        self.load_map=s.load_map
        self.map_path=s.map_path
        self.replace_length=s.replace_length
        self.bach_size=s.bach_size
        self.memory_num=s.memory_num
        self.memory=np.zeros((self.memory_num,self.n_feature*2+2))
        self.memory_counter=0
        self.learn_step_counter=0
        self.w_initializer = tf.random_normal_initializer(0.0, 0.3)
        self.b_initializer = tf.constant_initializer(0.1)
        #self.size = [self.n_feature, 10, 10, 10, self.n_action]
        self.size = [self.n_feature, 20, self.n_action]
        self.create_network()
        self.init()
        t_params=tf.get_collection("target_net_params")
        e_params=tf.get_collection("eval_net_params")
        self.replace_target_op=[tf.assign(t,e) for t,e in zip(t_params,e_params)]
    def init(self):
        self.sess = tf.Session()
        if self.load_map:
            pass
        else:
            self.sess.run(tf.global_variables_initializer())
        if self.save_graph:
            writer = tf.summary.FileWriter(self.graph_path, self.sess.graph)
            writer.close()
    def add_layer(self, i, input, input_size, output_size, c_name, name, activation_function=None):
        '''
        add new layer
        :i: layer ID
        :param input: input_data
        :param input_size: the length of input data
        :param output_size: the length of output data
        :param c_name: collections' name
        :param name: the name of the layer
        :param activation_function: activation function, ini is None
        :return: the output
        '''
        with tf.variable_scope(name):
            w = tf.get_variable("dqn_w"+str(i), [input_size, output_size], initializer=self.w_initializer, collections=c_name)
            b = tf.get_variable("dqn_b"+str(i), [1, output_size], initializer=self.b_initializer, collections=c_name)
            if activation_function is None:
                output = tf.matmul(input, w) + b
            else:
                output = activation_function(tf.matmul(input, w) + b)
        return output
    def add_network(self,input,c_name,name):
        '''
        Add network
        :param input: inpput data
        :param c_name: collection name
        :param name: the name of the network
        :return: the output data
        '''
        layer=input
        for i in range(len(self.size)-2):
            layer=self.add_layer(i+1,layer,self.size[i],self.size[i+1],c_name,self.name+"_"+name+"L"+str(i+1),tf.nn.relu)
        i = len(self.size)-1
        '''
        with tf.variable_scope(self.name+ "_" + name + 'Value'):
            w3 = tf.get_variable(self.name+ "_" + name + "w" + str(i), [self.size[-2], 1],
                                 initializer=self.w_initializer, collections=c_name)
            b3 = tf.get_variable(self.name+ "_" + name + "b" + str(i), [1, 1], initializer=self.b_initializer,
                                 collections=c_name)
            V = tf.matmul(layer, w3) + b3

        with tf.variable_scope(self.name+ "_" + name + 'Advantage'):
            w3 = tf.get_variable(self.name+ "_" + name + "w" + str(i), [self.size[-2], self.size[-1]],
                                 initializer=self.w_initializer, collections=c_name)
            b3 = tf.get_variable(self.name+ "_" + name + "b" + str(i), [1, self.size[-1]],
                                 initializer=self.b_initializer, collections=c_name)
            A = tf.matmul(layer, w3) + b3

        with tf.variable_scope('dueling_Q'):
            output = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)
        '''
        output=self.add_layer(10,layer,self.size[-2],self.size[-1],c_name,name+"Output")
        return output
    def create_network(self):
        '''
        Create the DRL network, including the eval network and the next network and the loss and the train
        :return: None
        '''
        self.q_target = tf.placeholder(tf.float32, [None, self.n_action], name="q_target")
        #Create eval network
        self.s = tf.placeholder(tf.float32, [None, self.n_feature], name="s")
        c_name = ["dqn_e_param", tf.GraphKeys.GLOBAL_VARIABLES]
        self.q_eval=self.add_network(self.s,c_name,"eval_")
        #Define the loss and train
        with tf.variable_scope("dqn_loss"):
            self.loss=tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope("dqn_train"):
            self._train_op=tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        #Create next network
        self.s_=self.s_ = tf.placeholder(tf.float32, [None, self.n_feature], name="s_")
        c_name= ["dqn_t_param", tf.GraphKeys.GLOBAL_VARIABLES]
        self.q_next=self.add_network(self.s_, c_name, "next_")
    def store_memory(self,s,a,r,s_):
        '''
        Store Memory
        :param s: the old state
        :param a: choosed actoion
        :param r: reward
        :param s_: the new state
        :return: None
        '''
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_num
        self.memory[index, :] = transition
        self.memory_counter += 1
        pass
    def chooes_action(self,state):
        '''
        Choose action according the state
        :param state: The state
        :return: The action ID, not the choosed action
        '''
        state=state[np.newaxis,:]
        if np.random.uniform()<self.e_greedy:
            action_value=self.sess.run(self.q_eval,feed_dict={self.s:state})
            action_ID=np.argmax(action_value)
        else:
            action_ID=np.random.randint(0,self.n_action)
        return action_ID
    def learn(self):
        if self.learn_step_counter % self.replace_length == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_num:
            sample_index = np.random.choice(self.memory_num, size=self.bach_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.bach_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_feature:],    # next observation
                       self.s: batch_memory[:, -self.n_feature:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_feature]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.bach_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_feature].astype(int)
        reward = batch_memory[:, self.n_feature + 1]

        max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_feature],
                                                self.q_target: q_target})
        '''
        if self.learn_step_counter % self.replace_length==0:
            self.sess.run(self.replace_target_op)
            print("\ntarget parameters replaced! \n")
        if self.memory_counter>self.memory_num:
            sample_index=np.random.choice(self.memory_num,size=self.bach_size)
        else:
            if self.memory_counter>self.bach_size:
                sample_index=np.random.choice(self.memory_counter,size=self.bach_size)
            else:
                sample_index=np.array([i for i in range(self.memory_counter)])
        batch_memory=self.memory[sample_index,:]
        q_next,q_eval=self.sess.run([self.q_next,self.q_eval],feed_dict={
            self.s_:batch_memory[:,   -self.n_feature:],
            self.s :batch_memory[:, : self.n_feature]
        })
        q_target=q_eval.copy()
        batch_index=np.arange(self.bach_size,dtype=np.int32)

        eval_act_index=batch_memory[:,self.n_feature].astype(int)
        reward=batch_memory[:,self.n_feature+1]
        
        
        q_target[batch_index,eval_act_index]=reward+self.gamma*np.max(q_next,axis=1)

        _,self.cost=self.sess.run([self._train_op,self.loss],
                                  feed_dict={
                                      self.s:batch_memory[:,:self.n_feature],
                                      self.q_target:q_target
                                  })
        '''
        if self.e_greedy<0.99:
            self.e_greedy+=self.e_greedy_increment
        self.learn_step_counter+=1
        return self.cost
if __name__=="__main__":
    DDQN=Double_Dueling_DQN()
    """
    s=np.array([10,0,0,10])
    for i in range(100):
        DDQN.store_memory(s,0,0,s)
    DDQN.chooes_action(s)
    DDQN.learn()
    """