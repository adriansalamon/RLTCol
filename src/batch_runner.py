import argparse
import os
import subprocess
import time
from datetime import datetime
from multiprocessing.dummy import Pool


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch of experiments usin the runner.py script"
    )
    parser.add_argument("policy", type=str, help="Policy to use")
    parser.add_argument("graph", type=str, help="Path to graph file")
    parser.add_argument("k", type=int, help="Number of colors")
    parser.add_argument("output_dir", type=str, help="Output directory")
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
        help="Time limit for run time of the algorithm, in seconds",
    )
    parser.add_argument(
        "--RL", type=str2bool, nargs="?", const=True, default=True, help="Use RL or not"
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        dest="num_jobs",
        default=4,
        help="Number of jobs to run in total",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        dest="concurrency",
        default=2,
        help="Number of jobs to run in parallel",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    null_file = open("/dev/null", "w")

    pool = Pool(args.concurrency)

    slack_ts = None
    big_time = time.time()

    def run_job(i):
        start_time = time.time()
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        print(f"Starting job {i + 1}")
        f = open(f"{args.output_dir}/{args.k}_{timestamp}.txt", "w")
        print(
            "running: ",
            f"python3 runner.py {args.policy} {args.graph} {args.k} -I {args.max_steps} -T {args.max_tabucol_iters} -B {args.beta} --time-limit {args.time_limit} --RL {args.RL}",
        )
        # Run in pool
        subprocess.run(
            f"python3 runner.py {args.policy} {args.graph} {args.k} -I {args.max_steps} -T {args.max_tabucol_iters} -B {args.beta} --time-limit {args.time_limit} --RL {args.RL}",
            shell=True,
            stdout=f,
        )
        end_time = time.time()

        print(f"Finished job {i + 1}", flush=True)

    result = pool.map_async(run_job, range(args.num_jobs))
    result.wait()

    big_end = time.time()
    print("All jobs finished", flush=True)
