import argparse
import glob
import os
import json


class Summarizer:
    results = {}

    def __init__(self):
        return

    def add_result(self, text):
        obj = json.loads(text)
        if obj["k"] not in self.results:
            self.results[obj["k"]] = []
        self.results[obj["k"]].append(obj)

    def summarize(self):
        for k, results in self.results.items():
            print(f"==================== k = {k} ====================")
            total_iters = 0
            total_time = 0
            succs = 0
            episodes = 0

            for result in results:
                if result["score"] == 0:
                    total_iters += result["iterations"]
                    total_time += result["time"]
                    episodes += result["episodes"]
                    succs += 1

            if len(results) == 0:
                print("No results found")
                return

            print(f"Graph: {results[0]['graph']}")
            print(f"Policy: {results[0]['policy']}")
            print(f"k: {results[0]['k']}")
            print(f"Success rate: {succs}/{len(results)}")
            if succs > 0:
                print(f"Average iters: {total_iters / succs}")
                print(f"Average time: {total_time / succs}")
                print(f"Average episodes: {int(episodes / succs)}")

            print("================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize results of RLTCol runs")
    parser.add_argument("folder", type=str, help="Path to result folder")

    args = parser.parse_args()

    result_files = glob.glob(os.path.join(args.folder, "*.txt"))
    summarizer = Summarizer()

    for result_file in result_files:
        with open(result_file, "r") as f:
            line = f.readline()
            while line:
                if line.strip() == "[DONE]":
                    next_line = f.readline()
                    summarizer.add_result(next_line)
                line = f.readline()

    summarizer.summarize()
