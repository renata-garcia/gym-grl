import gym
import time
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_speed=8
        self.steps=5
        self.max_torque=3.
        self.dt=.03
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta

        g = 9.81
        m = 0.055
        l = 0.042
        J = 0.000191
        b = 0.000003
        K = 0.0536
        R = 9.5
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        print("u: ", u, "\n")
        self.last_u = u # for rendering

        #INTEGRAR
        h = dt/self.steps #0.03 / 5

        th, thdot = self.state
        for i in range(0,self.steps):
            w = [1, 1/2, 1/2, 1]
            k = []
            states = []
            for j in range(0,4):
                print(j)
                th = th*w[j]
                thdot = thdot*w[j]
                
                newthdot = (1/J)*(m*g*l*np.sin(th)) -b*thdot - (K*K/R)*thdot + (K/R)*min(max(u, -3), 3)
                newth = th + newthdot*dt
                
                states.append((newth, newthdot))
                k.append(h* states[len(states)-1][1])
                print("states[j:", j, "]: ", states[j])
                print("states: ", states)
                print("k[0]: ", k[0])
                th = newth
                thdot = newthdot
            print("k[0]: ", k[0])
            print("k[0][0]: ", k[0][0])
            print("k[0][1]: ", k[0][1])
            print("w[0]: ", w[0])
            th = th + sum(w[j]*k[j][0] for j in range(0,4))/6
            thdot = thdot + sum(w[j]*k[j][1] for j in range(0,4))/6
            
        costs = -5*angle_normalize(th)**2 - .1*newthdot**2 - 1*(u**2) 
        #print("newth: ", newth)
        #print("costs: ", costs)
        #print("th: ", th)
        #print("newthdot: ", newthdot)
        #costs = costs * (newthdot - thdot) / 0.03

        self.state = np.array([newth, newthdot])
        return self._get_obs(), costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        #return np.array([np.cos(theta), np.sin(theta), thetadot])
        return np.array([theta, thetadot])

    def render(self, mode='human'):

        time.sleep(0.2)

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
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

