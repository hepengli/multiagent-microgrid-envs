from gym.envs.registration import register
from microgrid.environment_ieee34 import Environment_IEEE34
from microgrid.environment_ieee34_ddpg import Environment_IEEE34_DDPG
from microgrid.environment_cigre_mv import Environment_CIGRE_MV
from microgrid.environment_cigre_lv import Environment_CIGRE_LV

# Multiagent Microgrid envs
# ----------------------------------------

register(
    id='ieee34-v0',
    entry_point='microgrid:Environment_IEEE34',
    max_episode_steps=24,
)
register(
    id='ieee34ddpg-v0',
    entry_point='microgrid:Environment_IEEE34_DDPG',
    max_episode_steps=24,
)
register(
    id='CIGRE_MV-v0',
    entry_point='microgrid:Environment_CIGRE_MV',
    max_episode_steps=24,
)
register(
    id='CIGRE_LV-v0',
    entry_point='microgrid:Environment_CIGRE_LV',
    max_episode_steps=24,
)

