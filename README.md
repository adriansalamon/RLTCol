# Reinforcement learning driven TabuCol (RLTCol)

This code was written as part of our bachelor thesis titled: "Reinforcement learning for improved local search, applied to the graph coloring problem". This work was done at the [KTH Royal Institute of Technology](https://www.kth.se/en) in Stockholm, Sweden. This repository contains the code for the RLTCol algorithm, which is a hybrid heuristic algorithm for the graph coloring problem that uses reinforcement learning (RL).

The RLTCol algorithm works by iteratively running the local search algorithm TabuCol, and running an RL agent. The two components pass solutions to each other. The paper can be read [here](./paper/Paper.pdf).

## Code

The RL agent is implemented in Python using the [Tianshou](https://github.com/thu-ml/tianshou) library. TabuCol is implemented in Rust, using [maturin](https://github.com/PyO3/maturin) to interface with Python. The code is written for Python 3.10.

### Requirements

* Python 3.10
* A working Rust installation, see [here](https://www.rust-lang.org/tools/install) for instructions.

### Installation

Create a virtual environment and install the required Python packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, build the Rust code:

```bash
cd src/tabucol && maturin develop --release && cd -
```

## Usage

The graphs used as input need to be in the form of a DIMACS text file. The graphs used in the paper can be found [here](https://mat.tepper.cmu.edu/COLOR/instances.html). If they are in the compressed format, they can be decompressed using the translator found on the same page.

The source code for the RLTCol algorithm is located in the `src` directory. The TabuCol implementation in Rust is located in the `src/tabucol` directory.

### Training the RL agent

The RL agent can be trained using the `trainer.py` script. The script takes a number of arguments, which can be found by running `python trainer.py --help`.

The script will save the trained policy to the file specified by the `output` parameter. The policy can then be used to run the RLTCol algorithm.

### Running the RLTCol algorithm

The RLTCol algorithm can be run using the `runner.py` script. The script takes a number of arguments, which can be found by running `python runner.py --help`.

In order to run multiple jobs in parallel and/or in sequence, the `batch_runner.py` script can be used. The script takes a number of arguments, which can be found by running `python batch_runner.py --help`. This script will run the RLTCol algorithm and save the results to individual files in the directory specified by the `output_dir` parameter. The results can then be summarized using the `result_summarizer.py` script.

# Credits

The paper was written by Adrian Salamon (asalamon@kth.se) and Klara Sandström (klarasan@kth.se), supervised by [Stefano Markidis](https://www.kth.se/profile/markidis). 