import sys
import argparse
import subprocess
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
args, unknown = parser.parse_known_args()

args_launch = [
    "python",
    "-m",
    "torch.distributed.launch",
    "--nproc_per_node",
    str(args.world_size),
    "train.py",
    *unknown,
]

print(f"Launching distributed trianing on {args.world_size} gpus.")
p = subprocess.Popen(args_launch, stdout=sys.stdout, stderr=sys.stderr)

try:
    p.wait()
except KeyboardInterrupt:
    try:
        p.terminate()
    except OSError:
        pass
    p.wait()
