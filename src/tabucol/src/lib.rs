use ndarray::Array2;
use pyo3::prelude::*;
use rand::Rng;
use std::collections::HashSet;

#[pyclass]
pub struct TabuColSolver {
    graph: Vec<Vec<usize>>,
    solution: Vec<usize>,
    best_solution: Vec<usize>,
    best_score: i32,
    score: i32,
    delta: Array2<i32>,
    tabu: Array2<usize>,
    k: usize,
    n: usize,
    max_iters: usize,
    tabu_a: usize,
    tabu_alpha: f32,
    iters: usize,
    beta: f32,
}

#[pymethods]
impl TabuColSolver {
    #[new]
    #[pyo3(signature = (graph, k, max_iterations = 10000, tabu_a = 10, tabu_alpha = 1.2, beta = 0.2))]
    pub fn new(
        graph: Vec<Vec<usize>>,
        k: usize,
        max_iterations: usize,
        tabu_a: usize,
        tabu_alpha: f32,
        beta: f32,
    ) -> Self {
        let n = graph.len();

        let mut rng = rand::thread_rng();
        let mut solution = vec![0; n];
        for node in 0..n {
            solution[node] = rng.gen_range(0..k);
        }

        let mut solver = Self {
            graph,
            solution: solution.clone(),
            best_solution: solution,
            best_score: 0,
            score: 0,
            delta: Array2::zeros((n, k)),
            tabu: Array2::zeros((n, k)),
            k,
            n,
            max_iters: max_iterations,
            tabu_a,
            tabu_alpha,
            iters: 0,
            beta,
        };

        solver.initialize();

        solver
    }

    fn is_tabu(&self, node: usize, col: usize) -> bool {
        self.tabu[(node, col)] > self.iters
    }

    fn initialize(&mut self) {
        self.score = 0;
        self.delta = Array2::zeros((self.n, self.k));

        for node in 0..self.n {
            for neighbor in &self.graph[node] {
                if self.solution[node] == self.solution[*neighbor] {
                    self.score += 1;
                }
            }
            for col in 0..self.k {
                let mut res = 0;
                let node_color = self.solution[node];
                for neighbor in &self.graph[node] {
                    let i_cj = if self.solution[*neighbor] == col {
                        1
                    } else {
                        0
                    };
                    let i_ci = if self.solution[*neighbor] == node_color {
                        1
                    } else {
                        0
                    };
                    res += i_cj - i_ci;
                }
                self.delta[(node, col)] = res;
            }
        }

        self.score = self.score / 2;
        self.iters = 0;
        self.tabu = Array2::zeros((self.n, self.k));
        self.best_score = self.score;
    }

    pub fn set_solution(&mut self, solution: Vec<usize>) {
        self.solution = solution.clone();
        self.best_solution = solution.clone();

        self.initialize();
    }

    pub fn solve(&mut self) -> PyResult<(Vec<usize>, i32)> {
        let n = self.graph.len();
        let mut rng = rand::thread_rng();

        let initial_score = self.score;

        while self.iters < self.max_iters {
            // Find conflicting nodes

            let mut conflicting_nodes = HashSet::new();
            for node in 0..n {
                for neighbor in &self.graph[node] {
                    if self.solution[node] == self.solution[*neighbor] {
                        conflicting_nodes.insert(node);
                        break;
                    }
                }
            }
            if conflicting_nodes.len() == 0 {
                break;
            }

            // Find best move
            let mut best_moves: Vec<(usize, usize)> = Vec::new();
            let mut best_delta = i32::MAX;
            for node in conflicting_nodes {
                for col in 0..self.k {
                    if self.solution[node] == col {
                        continue;
                    }

                    let change = self.delta[(node, col)];

                    if !self.is_tabu(node, col) || self.score + change < self.best_score {
                        if change == best_delta {
                            best_moves.push((node, col));
                        }
                        if change <= best_delta {
                            best_moves = Vec::from([(node, col)]);
                            best_delta = change;
                        }
                    }
                }
            }

            if best_moves.len() == 0 {
                continue;
            }

            // Choose random move
            let move_index = rng.gen_range(0..best_moves.len());
            let (move_node, new_col) = best_moves[move_index];
            let old_col = self.solution[move_node];

            // Update delta matrix
            for neighbor in &self.graph[move_node] {
                let neighbor_col = self.solution[*neighbor];
                if neighbor_col != old_col {
                    self.delta[(*neighbor, old_col)] -= 1;
                }
                if neighbor_col != new_col {
                    self.delta[(*neighbor, new_col)] += 1;
                }
                if neighbor_col == new_col {
                    for col in 0..self.k {
                        if col != new_col {
                            self.delta[(*neighbor, col)] -= 1;
                        }
                        self.delta[(move_node, col)] -= 1;
                    }
                }
                if neighbor_col == old_col {
                    for col in 0..self.k {
                        if col != old_col {
                            self.delta[(*neighbor, col)] += 1;
                        }
                        self.delta[(move_node, col)] += 1;
                    }
                }
            }

            // Add move to tabu
            let tabu_time = rng.gen_range(0..self.tabu_a)
                + (self.tabu_alpha * self.score as f32).round() as usize;
            self.tabu[(move_node, self.solution[move_node])] = self.iters + tabu_time;

            // Update solution
            self.solution[move_node] = new_col;
            self.score += best_delta;
            if self.score <= self.best_score {
                self.best_score = self.score;
                self.best_solution = self.solution.clone();
            }

            if self.score == 0 {
                break;
            }

            // Iteration done
            self.iters += 1;
        }

        // e-greedy return
        if rng.gen::<f32>() < self.beta {
            return Ok((self.solution.clone(), initial_score - self.score));
        } else {
            return Ok((self.best_solution.clone(), initial_score - self.best_score));
        }
    }

    pub fn get_iters(&self) -> PyResult<usize> {
        Ok(self.iters)
    }

    pub fn get_best_score(&self) -> PyResult<i32> {
        Ok(self.best_score)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn tabucol(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TabuColSolver>()?;
    Ok(())
}
