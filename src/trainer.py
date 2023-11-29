import torch
from tianshou.utils.net.common import ActorCritic
from tianshou.data import VectorReplayBuffer, Collector
from tianshou.trainer import onpolicy_trainer
import gymnasium as gym
from network import ActorNetwork, CriticNetwork, GCPPPOPolicy
import networkx as nx
import gymnasium as gym
from gcp_env import GcpEnv
from tianshou.env import SubprocVectorEnv
import argparse
from gymnasium.envs.registration import register


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reinforcement learning based tabu search for graph coloring trainer"
    )
    parser.add_argument("output", type=str, help="Path to policy output file")
    parser.add_argument(
        "--input", type=str, default=None, help="Path to policy input file"
    )
    parser.add_argument(
        "-I",
        "--max-steps",
        type=int,
        dest="max_steps",
        default=300,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "-T",
        "--tabucol-iters",
        type=int,
        dest="tabucol_iters",
        default=5000,
        help="Number of iterations for tabucol",
    )
    parser.add_argument(
        "-E",
        "--epochs",
        type=int,
        dest="epochs",
        default=50,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "-N",
        "--nodes",
        type=int,
        dest="nodes",
        default=250,
        help="Number of nodes in training graphs",
    )
    parser.add_argument(
        "-P",
        "--probability",
        type=float,
        dest="probability",
        default=0.5,
        help="Probability of edge between nodes in training graph",
    )
    parser.add_argument(
        "-C",
        "--colors",
        type=int,
        dest="colors",
        default=24,
        help="Number of colors allowed when training",
    )

    tabucol_init = False

    args = parser.parse_args()

    nodes = args.nodes
    probability = args.probability
    colors = args.colors


    register(
        id="GcpEnvMaxIters-v0",
        entry_point="gcp_env:GcpEnv",
        max_episode_steps=args.max_steps,
    )
    spec = gym.spec("GcpEnvMaxIters-v0")

    env = gym.make(
        spec,
        graph=nx.gnp_random_graph(nodes, probability),
        k=colors,
        tabucol_init=tabucol_init,
    )

    print("Setting up environments...")

    train_envs = SubprocVectorEnv(
        [
            lambda: gym.make(
                spec,
                graph=nx.gnp_random_graph(nodes, probability),
                k=colors,
                tabucol_iters=args.tabucol_iters,
                tabucol_init=tabucol_init,
            )
            for _ in range(0, 10)
        ]
    )
    test_envs = SubprocVectorEnv(
        [
            lambda: gym.make(
                spec,
                graph=nx.gnp_random_graph(nodes, probability),
                k=colors,
                tabucol_iters=args.tabucol_iters,
                tabucol_init=tabucol_init,
            )
            for _ in range(0, 10)
        ]
    )

    print("Setting up policy...")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    node_features = 3
    col_features = 3

    print("Setting up networks...")
    actor = ActorNetwork(node_features, col_features, device=device).to(device)
    critic = CriticNetwork(node_features, col_features, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

    dist = torch.distributions.Categorical
    policy = GCPPPOPolicy(
        actor, critic, optim, dist, k=colors, nodes=nodes, action_space=env.action_space
    )

    if args.input is not None:
        print(f"Loading policy: {args.input}")
        policy.load_state_dict(torch.load(args.input, map_location=device))

    print("Setting up replay buffer and collectors...")

    replay_buffer = VectorReplayBuffer(30000, len(train_envs))
    train_collector = Collector(policy, train_envs, replay_buffer)
    test_collector = Collector(policy, test_envs)

    print("Training...")

    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.epochs,
        step_per_epoch=50000,
        repeat_per_collect=10,
        episode_per_test=10,
        batch_size=512,
        step_per_collect=3000,
    )

    torch.save(policy.state_dict(), args.output)
