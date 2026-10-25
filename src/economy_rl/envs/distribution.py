from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.stats import norm

from economy_rl.config import Config


def sample_normal_efficiency(n_agents: int, seed: int = 42) -> np.ndarray:
    """
    Sample efficiency z-scores from a normal curve for all agents such that:
    - At least 20% of the population have >= +1.0 sigma efficiency (high-skill labor)
    - At least 20% of the population have <= -1.0 sigma efficiency (laggards)
    - The remaining population receive efficiency values in (-1.0, 1.0) along the normal curve.
    """
    rng = np.random.default_rng(seed)
    k = max(1, int(np.ceil(0.20 * n_agents)))
    n_mid = n_agents - 2 * k

    # Standard normal CDF: Phi(-1.0) approx 0.158655, Phi(1.0) approx 0.841345
    p_neg_one = float(norm.cdf(-1.0))
    p_pos_one = float(norm.cdf(1.0))

    # Bottom >=20% (laggards, z <= -1.0)
    u_low = rng.uniform(0.001, p_neg_one, size=k)
    z_low = norm.ppf(u_low)

    # Top >=20% (high skill labor, z >= +1.0)
    u_high = rng.uniform(p_pos_one, 0.999, size=k)
    z_high = norm.ppf(u_high)

    # Middle population (-1.0 < z < 1.0)
    if n_mid > 0:
        u_mid = rng.uniform(p_neg_one, p_pos_one, size=n_mid)
        z_mid = norm.ppf(u_mid)
        z = np.concatenate([z_low, z_mid, z_high])
    else:
        z = np.concatenate([z_low, z_high])

    # Shuffle so agent indexing does not imply deterministic order before assignment
    rng.shuffle(z)
    return z.astype(np.float32)


def _wrap_components_for_effort(environment: Any) -> None:
    """
    Wraps simulation components so that the labor/effort cost to perform skills
    (moving, gathering, building) scales with each agent's individual effort multiplier,
    while nominal task rewards (coin, resource yields) remain identical for everyone.
    """
    for comp in environment.components:
        if getattr(comp, "_effort_wrapped", False):
            continue
        orig_step = comp.component_step

        def make_wrapper(c, orig):
            def wrapped_step():
                world = c.world
                pre_labor = {
                    a.idx: a.state["endogenous"]["Labor"]
                    for a in world.agents
                }
                orig()
                for a in world.agents:
                    delta = a.state["endogenous"]["Labor"] - pre_labor[a.idx]
                    if delta > 0:
                        mult = float(a.state.get("effort_multiplier", 1.0))
                        a.state["endogenous"]["Labor"] = pre_labor[a.idx] + delta * mult
            return wrapped_step

        comp.component_step = make_wrapper(comp, orig_step)
        comp._effort_wrapped = True


def apply_environment_distribution(
    environment: Any,
    config: Config,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Apply either uniform or non-uniform distribution to the environment:
    - Uniform distribution: resources scattered randomly, agents placed uniformly,
      all agents have identical baseline skills and identical effort (1.0x).
    - Non-uniform distribution:
      * Nominal rewards remain the same for all tasks (e.g. build_payment = 10.0 for everyone).
      * The effort (labor cost) it takes to perform skills changes according to efficiency sampled
        from a normal curve, with >= 20% having +1 sigma efficiency (low effort cost) and
        >= 20% having -1 sigma efficiency (high effort cost / laggards).
      * Spatial non-uniformity: efficient agents are close to resources, laggards are farther away.
    """
    _wrap_components_for_effort(environment)

    workers = [a for a in environment.all_agents if not getattr(a, "multi_action_mode", False)]
    n_workers = len(workers)
    if n_workers == 0:
        return environment.generate_observations()

    # Find Build component if registered
    build_component = next((c for c in environment.components if c.name == "Build"), None)

    # All tasks have the SAME base nominal reward across all agents
    STANDARD_BUILD_PAYMENT = float(getattr(config, "build_payment", 10.0))
    for worker in workers:
        worker.state["build_payment"] = STANDARD_BUILD_PAYMENT
        worker.state["build_skill"] = 1.0
        worker.state["bonus_gather_prob"] = 0.0
        if build_component is not None and hasattr(build_component, "sampled_skills"):
            build_component.sampled_skills[worker.idx] = 1.0

    if config.is_uniform_distribution:
        # Uniform Distribution: all agents have equal baseline effort (1.0x multiplier)
        for worker in workers:
            worker.state["efficiency"] = 0.0
            worker.state["effort_multiplier"] = 1.0
        return environment.generate_observations()

    # Non-Uniform Distribution:
    rng = np.random.default_rng(seed)

    # 1. Sample efficiency z-scores from normal curve
    z = sample_normal_efficiency(n_workers, seed=seed)

    # 2. Update agent effort multipliers based on efficiency
    # High skill (+1 sigma, z >= 1.0) -> requires less effort (multiplier <= 0.65)
    # Laggard (-1 sigma, z <= -1.0) -> requires more effort (multiplier >= 1.35)
    for i, worker in enumerate(workers):
        zi = float(z[i])
        effort_multiplier = float(np.clip(1.0 - 0.35 * zi, 0.25, 2.5))

        worker.state["efficiency"] = zi
        worker.state["effort_multiplier"] = effort_multiplier

    # 3. Spatial Non-Uniformity: Resource Proximity Allocation
    # Find all resource coordinates (Wood and Stone source blocks)
    wood = np.argwhere(environment.world.maps.get("WoodSourceBlock") > 0)
    stone = np.argwhere(environment.world.maps.get("StoneSourceBlock") > 0)
    if len(wood) > 0 and len(stone) > 0:
        resources = np.vstack([wood, stone])
    elif len(wood) > 0:
        resources = wood
    elif len(stone) > 0:
        resources = stone
    else:
        resources = np.zeros((0, 2), dtype=np.int32)

    if len(resources) > 0:
        H, W = environment.world_size
        accessible: List[Tuple[int, int, int]] = []
        sample_worker = workers[0]
        for r in range(H):
            for c in range(W):
                if environment.world.can_agent_occupy(r, c, sample_worker):
                    d = int(np.min(np.abs(resources[:, 0] - r) + np.abs(resources[:, 1] - c)))
                    accessible.append((r, c, d))

        if len(accessible) >= n_workers:
            accessible.sort(key=lambda item: item[2])
            distances = [item[2] for item in accessible]
            p25 = float(np.percentile(distances, 30))
            p75 = float(np.percentile(distances, 70))

            close_pool = [c for c in accessible if c[2] <= p25]
            far_pool = [c for c in accessible if c[2] >= p75]
            mid_pool = [c for c in accessible if p25 < c[2] < p75]

            rng.shuffle(close_pool)
            rng.shuffle(far_pool)
            rng.shuffle(mid_pool)

            environment.world.clear_agent_locs()
            occupied = set()

            # Place agents according to efficiency rank
            sorted_worker_indices = np.argsort(z)
            for idx in sorted_worker_indices:
                worker = workers[idx]
                zi = z[idx]
                if zi >= 1.0:
                    # High skill labor: placed close to resources
                    candidate_pool = close_pool + mid_pool + far_pool
                elif zi <= -1.0:
                    # Laggards: placed far from resources
                    candidate_pool = far_pool + mid_pool + close_pool
                else:
                    # Intermediate: placed in middle distance
                    candidate_pool = mid_pool + close_pool + far_pool

                for cell in candidate_pool:
                    r, c, d = cell
                    if (r, c) not in occupied and environment.world.can_agent_occupy(r, c, worker):
                        environment.world.set_agent_loc(worker, r, c)
                        occupied.add((r, c))
                        worker.state["initial_resource_distance"] = int(d)
                        break

    return environment.generate_observations()
