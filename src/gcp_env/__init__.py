from gymnasium.envs.registration import register
from .gcp_env import GcpEnv

register(
    id="GcpEnv-v0",
    entry_point="gcp_env:GcpEnv",
    max_episode_steps=1000,
)
