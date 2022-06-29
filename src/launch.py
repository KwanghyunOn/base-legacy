import sys
import argparse
import subprocess
import torch
from packaging.version import parse


parser = argparse.ArgumentParser()
parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
parser.add_argument("--script", type=str, default="train.py")
parser.add_argument("--log_dir", type=str, default="./temp/log/")
parser.add_argument("--redirect", action="store_true")
args, unknown = parser.parse_known_args()

if parse(torch.__version__) >= parse("1.10"):
    """
    Redirect std streams of all workers into a log file except the worker with rank 0. Only std
    streams from rank 0 will be printed to the console. Set --log_dir in args_launch to set
    directory for log files.
    """
    if args.redirect:
        redirects = ",".join(f"{rank}:3" for rank in range(1, args.world_size))
    else:
        redirects = "0"

    args_launch = [
        "torchrun",
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "localhost:0",
        "--nproc_per_node",
        str(args.world_size),
        "--log_dir",
        args.log_dir,
        "--redirects",
        redirects,
        args.script,
        "--distributed",
        *unknown,
    ]
else:
    args_launch = [
        "python",
        "-m",
        "torch.distributed.launch",
        "--use_env",
        "--nproc_per_node",
        str(args.world_size),
        "train.py",
        "--distributed",
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
