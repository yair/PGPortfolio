import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

print ("Registering environment 'GymPGP-v0'")

register(
    id='GymPGP-v0',
    entry_point='gym_pgp.envs:PGPEnv',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic = False,
)

