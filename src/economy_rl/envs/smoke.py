from economy_rl.config import Config
from economy_rl.envs.adapter import encode_actions, inspect_agents
from economy_rl.envs.parallel import ParallelWorlds


def run_smoke_test() -> None:
    config = Config()
    worlds = ParallelWorlds(config)
    try:
        planner_id, worker_ids, planner_dims, worker_action_size = inspect_agents(worlds.worlds[0].environment)
        planner_action = [0 for _ in planner_dims]
        worker_actions = [0 for _ in worker_ids]
        observations, rewards, done, _ = worlds.step(
            0, encode_actions(planner_id, worker_ids, planner_action, worker_actions)
        )
        print("environment: ready")
        print("world_size:", config.world_size)
        print("workers:", len(worker_ids))
        print("worlds:", config.n_worlds)
        print("planner_action_dims:", planner_dims)
        print("worker_action_size:", worker_action_size)
        print("done:", done)
        print("reward_keys:", sorted(str(key) for key in rewards))
        print("observation_agents:", len(observations))
    finally:
        worlds.close()


if __name__ == "__main__":
    run_smoke_test()
