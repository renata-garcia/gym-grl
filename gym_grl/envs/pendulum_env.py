from decimal import * 
import gym
import time
import math
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import sys
import time
sys.path.append('../../grl/build')
import grlpy

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        #GRL load configuration
        self.conf = grlpy.Configurator("../../grl/cfg/pendulum/pendulum.yaml")
        #GRL instantiate configuration (construct objects)
        self.inst = self.conf.instantiate()
        self.env = grlpy.Environment(self.inst["environment"])
        
        self.max_speed = 100000 #TODO observation_max
        #min_torque = float(str(self.inst["experiment"]["environment"]["task"]["action_min"][0]))
        #max_torque = float(str(self.inst["experiment"]["environment"]["task"]["action_max"][0]))
        self.min_torque = -3
        self.max_torque = 3
        #float(str(self.inst["experiment"]["environment"]["task"]["dt"]))
        self.dt = .03
        self.viewer = None

        # self.obs_dims = float(str(self.inst["environment"]["task"]["observation_dims"]))
        self.obs_dims = 3
        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=self.min_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high,dtype=np.float32)
        
        self.terminal = 0

    def step(self,u):
        (self.obs, reward, self.terminal) = self.env.step(u)
        self.last_u = u
        # print("pendulum:step:state: ", self.state, ", reward: ", reward, ", self._get_obs()", self._get_obs())
        return self.obs, reward, False, {}

    def reset(self):
        #GRL Restart environment and agent
        # self.state = self.env.start(0)
        self.last_u = 0 #self.agent.start(self.obs)
        self.obs = self.env.start(0)
        # print("pendulum:reset:state: ", self.state, ", self._get_obs(): ", self._get_obs())
        return self.obs

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        #print("pendulum: state[0]: ", self.state[0], " - +pi/2", np.pi/2)
        self.pole_transform.set_rotation(math.asin(self.obs[0]) - np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

