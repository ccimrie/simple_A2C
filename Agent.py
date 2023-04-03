import numpy as np
from A2C import A2C
import time

class Agent(object):

    def __init__(self):
        self.rl=A2C(2,4)

        ## Agent params
        self.x=0
        self.y=0
        self.goal_x=0
        self.goal_y=0
        self.vel_max=2.0
        self.f=open('test.txt', 'w+')

    def checkVelMag(self,act):
        ## Limit velocity components
        vel_mag=act[0]**2+act[1]**2
        if (vel_mag)>self.vel_max**2:
            factor=self.vel_max/np.sqrt(vel_mag)
            act[0]=act[0]*factor
            act[1]=act[1]*factor
        return act

    def move(self, act):
        ## move agent
        #print(act)
        act=self.checkVelMag(np.clip(act*2.0,-2,2))
        self.x=self.x+act[0]
        self.y=self.y+act[1]

    def getX(self):
        return self.x

    def getY(self):
        return self.y
 
    def episode(self, TT):
        thresh=0.005
        for t in np.arange(TT):
            state=np.array([self.getX(), self.getY(), self.goal_x, self.goal_y])
            ## Take step in the world
            act=self.rl.step(state/5.0)
            self.move(act)
            ## Get reward
            dist=(self.getX()-self.goal_x)**2+(self.getY()-self.goal_y)**2
            reward=1.0/(np.sqrt(dist)+1e-2)
            self.rl.recordReward(reward)
            ## Learn
            self.rl.learnActor()
            self.rl.learnCritic()
            self.rl.clearHistory()
           if np.sqrt(dist)<thresh:
               print("REACHED: ", np.sqrt(dist))
               break
        dist=(self.getX()-self.goal_x)**2+(self.getY()-self.goal_y)**2
        reward=1.0/(np.sqrt(dist)+1e-2)
        self.f.write(str(reward))
        self.f.write('\n')
        self.f.flush()


    def act(self):
        thresh=0.005
        if np.sqrt(dist)<thresh:
           print("REACHED: ", np.sqrt(dist))
           break

        state=np.array([self.getX(), self.getY(), self.goal_x, self.goal_y])

        ## Take step in the world
        act=self.rl.act(state).copy()
        self.move(act)

        ## Get reward
        dist=(self.getX()-self.goal_x)**2+(self.getY()-self.goal_y)**2
        reward=1.0/(np.sqrt(dist)+1e-1)
        print(reward)

    def reset(self):
        ## Reset agent and goal position
        self.x=np.random.uniform(-5,5)
        self.y=np.random.uniform(-5,5)

        self.goal_x=np.random.uniform(-5,5)
        self.goal_y=np.random.uniform(-5,5)
        self.rl.clearHistory()

