from unittest import TestCase
import gymnasium as gym
import networkx as nx
import numpy as np

import sys
import os.path

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import gcp_env


class TestEnv(TestCase):
    def test_reset_works(self):
        G = nx.from_edgelist([(0, 1), (0, 2), (1, 2)])
        initial_solution = np.array([0, 1, 1])

        env = gym.make("GcpEnv-v0", graph=G, k=2)
        obs, _ = env.reset(options={"initial_solution": initial_solution})

        np.testing.assert_array_equal(
            obs["node_features"], np.array([[2, 0, 1], [2, 1, 2], [2, 1, 2]])
        )
        np.testing.assert_array_equal(
            obs["col_features"],
            np.array(
                [[[0, 1, 3], [2, 2, 3]], [[1, 1, 3], [1, 2, 3]], [[1, 1, 3], [1, 2, 3]]]
            ),
        )

    def test_step_works(self):
        G = nx.from_edgelist([(0, 1), (0, 2), (1, 2)])
        initial_solution = np.array([0, 1, 1])

        env = gym.make("GcpEnv-v0", graph=G, k=2)
        env.reset(options={"initial_solution": initial_solution})
        res = env.step(np.array([1, 0]))
        obs = res[0]
        reward = res[1]
        np.testing.assert_array_equal(
            obs["node_features"], np.array([[2, 1, 2], [2, 1, 2], [2, 0, 1]])
        )
        np.testing.assert_array_equal(
            obs["col_features"],
            np.array(
                [[[1, 2, 3], [1, 1, 3]], [[1, 2, 3], [1, 1, 3]], [[2, 2, 3], [0, 1, 3]]]
            ),
        )
        self.assertEqual(reward, 0)

    def test_step_works_v2(self):
        G = nx.from_edgelist([(0, 1), (1, 2), (1, 3)])
        initial_solution = np.array([1, 1, 0, 0])

        env = gym.make("GcpEnv-v0", graph=G, k=2)
        env.reset(options={"initial_solution": initial_solution})
        res = env.step(np.array([1, 0]))
        obs, rew = res[0:2]
        np.testing.assert_array_equal(
            obs["node_features"], np.array([[1, 0, 1], [3, 2, 2], [1, 1, 1], [1, 1, 1]])
        )
        np.testing.assert_array_equal(
            obs["col_features"],
            np.array(
                [
                    [[1, 3, 4], [0, 1, 4]],
                    [[2, 3, 4], [1, 1, 4]],
                    [[1, 3, 4], [0, 1, 4]],
                    [[1, 3, 4], [0, 1, 4]],
                ]
            ),
        )
        self.assertEqual(rew, -1)

    def test_step_works_with_same_as_prev(self):
        G = nx.from_edgelist([(0, 1), (0, 2), (1, 2)])
        initial_solution = np.array([0, 1, 1])

        env = gym.make("GcpEnv-v0", graph=G, k=2)
        env.reset(options={"initial_solution": initial_solution})

        res = env.step(np.array([2, 1]))
        obs = res[0]

        np.testing.assert_array_equal(
            obs["node_features"], np.array([[2, 0, 1], [2, 1, 2], [2, 1, 2]])
        )
        np.testing.assert_array_equal(
            obs["col_features"],
            np.array(
                [[[0, 1, 3], [2, 2, 3]], [[1, 1, 3], [1, 2, 3]], [[1, 1, 3], [1, 2, 3]]]
            ),
        )

    def test_step_works_with_new_col(self):
        G = nx.from_edgelist([(0, 1), (0, 2), (1, 2)])
        initial_solution = np.array([0, 0, 0])

        env = gym.make("GcpEnv-v0", graph=G, k=2)
        env.reset(options={"initial_solution": initial_solution})

        res = env.step(np.array([2, 1]))
        obs = res[0]

        np.testing.assert_array_equal(
            obs["node_features"], np.array([[2, 1, 2], [2, 1, 2], [2, 0, 1]])
        )
        np.testing.assert_array_equal(
            obs["col_features"],
            np.array(
                [[[1, 2, 3], [1, 1, 3]], [[1, 2, 3], [1, 1, 3]], [[2, 2, 3], [0, 1, 3]]]
            ),
        )
