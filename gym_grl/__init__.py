import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Pendulum-grl-v0',
    entry_point='gym_grl.envs:PendulumEnv',
    timestep_limit=100,
    reward_threshold=1.0,
    nondeterministic = True,
)

