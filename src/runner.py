import numpy as np
from gcp_env import GcpEnv
from gymnasium.envs.registration import register
import argparse
import random
import networkx as nx
import time
import json
import os
import sys

import torch
import gymnasium as gym
from tianshou.data import Collector
from tianshou.utils.net.common import ActorCritic
from tianshou.env import DummyVectorEnv
from network import ActorNetwork, CriticNetwork, GCPPPOPolicy
from tianshou.policy import BasePolicy
from tianshou.data import Batch


class ThinGraph(nx.Graph):
    all_edge_dict = {"weight": 1}

    def single_dict(self):
        return self.all_edge_dict

    edge_attr_dict_factory = single_dict
    node_attr_dict_factory = single_dict


def read_graph_from_file(path):
    graph = nx.Graph()
    with open(path, "r") as f:
        for line in f.readlines():
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                n, m = line.split()[2:4]
                n, m = int(n), int(m)

                graph.add_nodes_from(range(n))
                continue
            if line.startswith("e"):
                u, v = line.split()[1:3]
                u, v = int(u), int(v)
                graph.add_edge(u - 1, v - 1)
                continue
    return graph


# Random initial solution


def initial_solution(graph, k):
    solution = {}
    for node in graph.nodes():
        solution[node] = random.randint(0, k - 1)
    return solution


def calculate_score(graph, solution):
    score = 0
    for node in graph.nodes():
        for neighbor in graph.neighbors(node):
            if solution[node] == solution[neighbor]:
                score += 1
    score = score // 2
    return score


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class RandomGCPPolicy(BasePolicy):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def forward(self, batch, state=None, **kwargs):
        action = np.array([self.action_space.sample()])
        return Batch(act=action)

    def learn(self, batch: Batch, **kwargs):
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reinforcement learning based tabu search for graph coloring"
    )
    parser.add_argument("policy", type=str, help="Path to policy to use")
    parser.add_argument("graph", type=str, help="Path to graph file")
    parser.add_argument("k", type=int, help="Number of colors")
    parser.add_argument(
        "-I",
        "--max-steps",
        type=int,
        dest="max_steps",
        default=1000,
        help="Max RL steps per episode",
    )
    parser.add_argument(
        "-T",
        "--max-tabucol-iters",
        type=int,
        dest="max_tabucol_iters",
        default=100000,
        help="Max tabucol iterations in each episode",
    )
    parser.add_argument(
        "-E",
        "--max-episodes",
        type=int,
        dest="episodes",
        default=None,
        help="Max episodes to run",
    )
    parser.add_argument(
        "-B",
        "--beta",
        type=float,
        dest="beta",
        default=0.2,
        help="Beta parameter in RLTCol",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        dest="time_limit",
        default=1000,
        help="Time limit for run time ot the algorithm, in seconds",
    )
    parser.add_argument(
        "--RL", type=str2bool, nargs="?", const=True, default=True, help="Use RL or not"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    node_features = 3
    col_features = 3

    args = parser.parse_args()
    graph = read_graph_from_file(args.graph)

    register(
        id="GcpEnvMaxIters-v0",
        entry_point="gcp_env:GcpEnv",
        max_episode_steps=args.max_steps,
    )

    env = gym.make(
        "GcpEnvMaxIters-v0",
        graph=graph,
        k=args.k,
        tabucol_iters=args.max_tabucol_iters,
        beta=args.beta,
    )
    vector_env = DummyVectorEnv([lambda: env])
    policy = None

    if args.RL:
        actor = ActorNetwork(node_features, col_features, device=device).to(device)
        critic = CriticNetwork(node_features, col_features, device=device).to(device)
        actor_critic = ActorCritic(actor, critic).to(device)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

        dist = torch.distributions.Categorical
        policy = GCPPPOPolicy(
            actor,
            critic,
            optim,
            dist,
            k=args.k,
            nodes=len(graph),
            action_space=env.action_space,
        ).to(device)

        policy.load_state_dict(torch.load(args.policy, map_location=device))
    else:
        # Use random policy if RL is not used
        policy = RandomGCPPolicy(action_space=env.action_space)

    policy.eval()
    eval_collector = Collector(policy, vector_env)
    eval_collector.reset()

    print("-------------------------")
    print("Starting evaluation, info:")
    print(f"Device: {device}")
    print(f"Policy: {args.policy}")
    print(f"Graph: {args.graph}")
    print(f"k: {args.k}")
    print(f"Beta: {args.beta}")
    print(f"Max steps: {args.max_steps}")
    print(f"Max tabucol iters: {args.max_tabucol_iters}")
    print(f"Max episodes: {args.episodes}")
    print(f"Time limit: {args.time_limit}")
    print("-------------------------")
    sys.stdout.flush()

    episodes = 0
    tot_start = time.time()

    while True:
        if args.episodes is not None and episodes >= args.episodes:
            break

        run_time = time.time() - tot_start
        if run_time > args.time_limit:
            break

        episodes += 1
        start_time = time.time()

        result = eval_collector.collect(n_episode=1)

        end_time = time.time()
        graph = env.env.get_graph()
        solution = env.env.get_solution()
        best_score = calculate_score(graph, solution)

        print(
            f"Episode done, score: {best_score}, time: {end_time - start_time}, result: {result}",
            flush=True,
        )
        if best_score == 0:
            break

    tot_end = time.time()
    iterations = episodes * (args.max_steps + args.max_tabucol_iters)

    print("[DONE]")
    json_result = {
        "score": best_score,
        "episodes": episodes,
        "iterations": iterations,
        "time": tot_end - tot_start,
        "time_per_iteration": (tot_end - tot_start) / iterations,
        "solution": solution,
        "graph": args.graph,
        "k": args.k,
        "max_steps": args.max_steps,
        "max_tabucol_iters": args.max_tabucol_iters,
        "policy": args.policy,
        "RL": args.RL,
    }

    # JSONIFY
    print(json.dumps(json_result, default=str), flush=True)

    # exit process
    os._exit(0)
