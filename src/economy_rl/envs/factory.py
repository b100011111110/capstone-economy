import types
from typing import Any, Dict
import numpy as np

from ai_economist import foundation

from economy_rl.config import Config


def _apply_fungible_build_patch(build_component: Any, cost: int = 2) -> None:
    """Treat Wood and Stone as fungible building resources requiring `cost` total units."""
    def fungible_agent_can_build(self, agent):
        inv = agent.state["inventory"]
        total = float(inv.get("Wood", 0.0)) + float(inv.get("Stone", 0.0))
        if total < cost:
            return False
        if self.world.location_resources(*agent.loc):
            return False
        if self.world.location_landmarks(*agent.loc):
            return False
        return True

    def fungible_component_step(self):
        world = self.world
        build = []
        for agent in world.get_random_order_agents():
            action = agent.get_component_action(self.name)
            if action is None or action == 0:
                continue
            elif action == 1:
                if self.agent_can_build(agent):
                    remaining = float(cost)
                    for res in ["Wood", "Stone"]:
                        curr = float(agent.state["inventory"].get(res, 0.0))
                        deduct = min(curr, remaining)
                        agent.state["inventory"][res] -= deduct
                        remaining -= deduct
                        if remaining <= 0:
                            break
                    loc_r, loc_c = agent.loc
                    world.create_landmark("House", loc_r, loc_c, agent.idx)
                    agent.state["inventory"]["Coin"] += agent.state["build_payment"]
                    agent.state["endogenous"]["Labor"] += self.build_labor
                    agent.houses_built = getattr(agent, "houses_built", 0.0) + 1.0
                    build.append({
                        "builder": agent.idx,
                        "loc": np.array(agent.loc),
                        "income": float(agent.state["build_payment"]),
                    })
            else:
                raise ValueError
        self.builds.append(build)

    build_component.agent_can_build = types.MethodType(fungible_agent_can_build, build_component)
    build_component.component_step = types.MethodType(fungible_component_step, build_component)


def make_environment(config: Config, seed: int = 0) -> Any:
    """Create one Foundation environment with the requested economic scale."""
    components_spec = []
    for c in config.components:
        if isinstance(c, str):
            if c == "Build" and hasattr(config, "build_payment") and config.build_payment is not None:
                components_spec.append({"Build": {"payment": float(config.build_payment)}})
            else:
                components_spec.append({c: {}})
        elif isinstance(c, dict):
            components_spec.append(c)

    environment_config: Dict[str, Any] = {
        "scenario_name": config.scenario_name,
        "components": components_spec,
        "world_size": config.world_size,
        "n_agents": config.n_agents,
        "episode_length": config.episode_length,
    }
    environment = foundation.make_env_instance(**environment_config)

    if getattr(config, "fungible_build_resources", False):
        cost = int(getattr(config, "build_resource_cost", 2) or 2)
        for comp in environment.components:
            if comp.name == "Build":
                _apply_fungible_build_patch(comp, cost=cost)

    if hasattr(environment, "seed"):
        environment.seed(seed)
    return environment
