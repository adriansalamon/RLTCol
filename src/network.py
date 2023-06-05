from torch import nn
import torch
from tianshou.policy.modelfree.ppo import PPOPolicy
from typing import Any, Optional, Union, Type
from tianshou.data import Batch
import numpy as np


class NodeNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


class ColNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


# Assumes that we are using a batch size equal to the number of nodes,
# returns a tensor n x k tensor containing the probabilities of choosing an action
class ActorNetwork(nn.Module):
    def __init__(self, node_features, col_features, device="cpu") -> None:
        super().__init__()
        self.node_model = nn.Sequential(
            NodeNetwork(node_features),
            nn.Softmax(dim=1),
        )

        self.col_model = nn.Sequential(
            ColNetwork(col_features),
            nn.Softmax(dim=2),
        )
        self.device = device

    def forward(self, obs, state=None, info={}):
        if self.device is not None:
            obs["node_features"] = torch.as_tensor(
                obs["node_features"], device=self.device, dtype=torch.float32
            )
            obs["col_features"] = torch.as_tensor(
                obs["col_features"], device=self.device, dtype=torch.float32
            )

        node_probs = torch.squeeze(self.node_model(obs["node_features"]), -1)
        col_probs = torch.squeeze(self.col_model(obs["col_features"]), -1)

        return (
            torch.flatten(
                torch.transpose(
                    (torch.transpose(col_probs, -1, -2) * node_probs[:, None]), -1, -2
                ),
                start_dim=1,
            ),
            state,
        )


class CriticNetwork(nn.Module):
    def __init__(self, node_features, col_features, device="cpu") -> None:
        super().__init__()
        self.node_model = nn.Sequential(
            NodeNetwork(node_features),
        )

        self.col_model = nn.Sequential(
            ColNetwork(col_features),
        )

        self.out_layer = nn.Sequential(nn.LazyLinear(1))
        self.device = device

    def forward(self, obs, **kwargs):
        if self.device is not None:
            obs["node_features"] = torch.as_tensor(
                obs["node_features"], device=self.device, dtype=torch.float32
            )
            obs["col_features"] = torch.as_tensor(
                obs["col_features"], device=self.device, dtype=torch.float32
            )

        node_probs = torch.squeeze(self.node_model(obs["node_features"]), -1)
        col_probs = torch.squeeze(self.col_model(obs["col_features"]), -1)

        last_inputs = torch.flatten(
            torch.transpose(
                (torch.transpose(col_probs, -1, -2) * node_probs[:, None]), -1, -2
            ),
            start_dim=1,
        )
        return self.out_layer(last_inputs)


class GCPPPOPolicy(PPOPolicy):
    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        nodes: int,
        k: int,
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            actor,
            critic,
            optim,
            dist_fn,
            eps_clip,
            dual_clip,
            value_clip,
            advantage_normalization,
            recompute_advantage,
            **kwargs
        )

        self.k = k
        self.n = nodes

    def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
        def mapper(x):
            node, col = divmod(x, self.k)
            return np.array([node, col])

        res = np.array([mapper(xi) for xi in act])
        return res
