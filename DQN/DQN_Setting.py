from Env.env import env
env=env()
state=env.reset()
class DQN_Setting():
    def __init__(self):
        #self.n_action=len(env.actions)#Action Number
        self.n_action=2#Action Number
        #self.n_feature=len(state) #Feature length
        self.n_feature=4 #Feature length
        self.memory_num=10000 #stored memory number
        self.learning_rate=0.9 #learning rate
        self.gamma=0.8 #discount rate
        self.e_greedy=0.0 #greedy rete
        self.e_greedy_increment=0.001 #greedy increce rate
        self.bach_size=500 #the number of memory choosed for learning
        self.replace_length=50 #the number of steps of exchange network
        self.output_graph=True #whether to save the graph
        self.graph_path="./log"# The path to save the graph
        self.load_map=False #decide whether load the excisting map
        self.map_path="./map" #the path of the map,
if __name__=="__main__":
    s=DQN_Setting()
    print(s.n_action)
    print(s.n_feature)
    print(state)