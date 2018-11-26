from decimal import * 
import gym
import time
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

        #GRL Get reference to agent and environment
        #self.agent = grlpy.Agent(self.inst["experiment"]["agent"])
        self.env = grlpy.Environment(self.inst["environment"])
        max_speed = 100000 #TODO observation_max
        obs_dims = float(str(self.inst["environment"]["task"]["observation_dims"]))
        print("pendulum-grl:obs_dims: ", obs_dims)
        if (obs_dims == 2):
            high = np.array([np.pi, max_speed])
        else:
            high = np.array([1., 1., self.max_speed])

        self.observation_space = spaces.Box(low=-high, high=high,dtype=np.float32)
        #min_torque = float(str(self.inst["experiment"]["environment"]["task"]["action_min"][0]))
        #max_torque = float(str(self.inst["experiment"]["environment"]["task"]["action_max"][0]))
        min_torque = -3
        max_torque = 3
        self.action_space = spaces.Box(low=min_torque, high=max_torque, shape=(1,), dtype=np.float32)
        #self.observation_space = spaces.Box(low=self.inst["experiment"]["environment"]["task"]["action_min"], high=self.inst["experiment"]["environment"]["task"]["action_max"], dtype=np.float32)

        self.terminal = 0

        self.viewer = None

    def step(self,u):
        self.last_u = u
        (self.obs, reward, self.terminal) = self.env.step(u)
        self.state = self.obs
        #self.action = self.agent.step(self.obs, reward)
        return self.obs, reward, False, {}

    def reset(self):
        #GRL Restart environment and agent
        self.obs = self.env.start(0)
        self.last_u = 0 #self.agent.start(self.obs)
        self.state = self.obs
        return self.obs

    def _get_obs(self):
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
        self.pole_transform.set_rotation(self.state[0] - np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

