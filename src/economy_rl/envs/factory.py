from typing import Any, Dict

from ai_economist import foundation

from economy_rl.config import Config


def make_environment(config: Config, seed: int = 0) -> Any:
    """Create one Foundation environment with the requested economic scale."""
    environment_config: Dict[str, Any] = {
        "scenario_name": config.scenario_name,
        "components": [{name: {}} for name in config.components],
        "world_size": config.world_size,
        "n_agents": config.n_agents,
        "episode_length": config.episode_length,
    }
    environment = foundation.make_env_instance(**environment_config)
    if hasattr(environment, "seed"):
        environment.seed(seed)
    return environment
