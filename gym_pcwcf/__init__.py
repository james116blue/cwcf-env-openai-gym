from gym.envs.registration import register

register(
    id='pcwcf-v0',
    entry_point='gym_pcwcf.envs:DiscretePcwcfEnv',
    )
register(
    id='cwcf-v0',
    entry_point='gym_pcwcf.envs:CwcfEnv'
    )