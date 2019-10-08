import numpy as np
from Env.take_reward import *
from Env.env_setting import Settings
s=Settings()
class env():
    '''
    ACC enviroment Class
    '''
    def __init__(self,ego_x=s.aim_x,ego_v=s.ego_v,ego_a=s.ego_a,aim_x=s.aim_x,aim_v=s.aim_v,aim_a=s.aim_a):
        self.i_ego_x=ego_x
        self.i_ego_v=ego_v
        self.i_ego_a=ego_a
        self.i_aim_x=aim_x
        self.i_aim_v=aim_v
        self.i_aim_a=aim_a
        self.actions=[-1,0,1]
        self.t=0.1
        self.action=0
        self.reset()
    def reset(self):
        self.ego_x=self.i_ego_x
        self.ego_v=self.i_ego_v
        self.ego_a=self.i_ego_a
        self.aim_x=self.i_aim_x
        self.aim_v=self.i_aim_v
        self.aim_a=self.i_aim_a
        state=self.make_state(action=0,goal=0)
        return state
    def calculate(self,x,v,a):
        X=x+v*self.t+0.5*a*self.t
        V=v+self.t*a
        return X,V
    def aim_a_change(self):
        '''
        aim_car's Acceleration
        :return:
        '''
        self.aim_a=0#ini=0
        pass
    def make_state(self,action,goal):
        '''

        :param action: The choosed Action(sub-action)
        :param goal: The choosed Goal(meta-action)
        :return: state=[dis,delat_v,delta_a,ego_v,ego_a,action,delta_dis]
        '''
        gap_x=self.aim_x-self.ego_x
        gap_v=self.aim_v-self.ego_v
        gap_a=self.aim_a-self.ego_a
        d=abs(gap_x-goal)
        state=[gap_x,gap_v,gap_a,self.ego_v,self.ego_a,action,d]
        return np.array(state)
    def make_feature(self,gap_x,gap_v,gap_a,action,d):
        '''
        Make The Feature for take the reward
        :return:
        '''
        feature=[gap_x,gap_v,gap_a,self.ego_v,self.ego_a,action-self.action,d]
        return feature
        pass
    def step(self,action,goal):
        '''
        :param action: ego_car's Acceleration
        :return: state,reward,done
        '''
        #Set Accelerations
        self.ego_a=action
        self.aim_a_change()
        #Calculate x and v
        self.ego_x,self.ego_v=self.calculate(self.ego_x,self.ego_v,self.ego_a)
        self.aim_x,self.aim_v=self.calculate(self.aim_x,self.aim_v,self.aim_a)
        #make new state
        state=self.make_state(action,goal)
        #take reward ,which is the map bettwen feature and reward
        deature=self.make_feature(state[0],state[1],state[2],action,state[-1])
        reward=take_reward(deature)
        self.action=action
        #Judge the terminal state
        done_=done(deature)
        return state,reward,done_
if __name__=="__main__":
    env=env()
    print(env.ego_x)