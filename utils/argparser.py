import argparse

from ray.tune.config_parser import make_parser
from ray.tune.result import DEFAULT_RESULTS_DIR

EXAMPLE_USAGE = """
Training example:
    python ./train.py --run DQN --env CartPole-v0 --no-log-flatland-stats

Training with Config:
    python ./train.py -f experiments/flatland_random_sparse_small/global_obs/ppo.yaml


Note that -f overrides all other trial-specific command-line options.
"""


def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE)

    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--ray-address",
        default=None,
        type=str,
        help="Connect to an existing Ray cluster at this address instead "
             "of starting a new one.")
    parser.add_argument(
        "--ray-num-cpus",
        default=None,
        type=int,
        help="--num-cpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-gpus",
        default=None,
        type=int,
        help="--num-gpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-nodes",
        default=None,
        type=int,
        help="Emulate multiple cluster nodes for debugging.")
    parser.add_argument(
        "--ray-redis-max-memory",
        default=None,
        type=int,
        help="--redis-max-memory to use if starting a new cluster.")
    parser.add_argument(
        "--ray-memory",
        default=None,
        type=int,
        help="--memory to use if starting a new cluster.")
    parser.add_argument(
        "--ray-object-store-memory",
        default=None,
        type=int,
        help="--object-store-memory to use if starting a new cluster.")
    parser.add_argument(
        "--experiment-name",
        default="default",
        type=str,
        help="Name of the subdirectory under `local_dir` to put results in.")
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_RESULTS_DIR,
        type=str,
        help="Local dir to save training results to. Defaults to '{}'.".format(
            DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--upload-dir",
        default="",
        type=str,
        help="Optional URI to sync training results to (e.g. s3://bucket).")
    parser.add_argument(
        "-v", action="store_true", help="Whether to use INFO level logging.")
    parser.add_argument(
        "-vv", action="store_true", help="Whether to use DEBUG level logging.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume previous Tune experiments.")
    parser.add_argument(
        "--torch",
        action="store_true",
        help="Whether to use PyTorch (instead of tf) as the DL framework.")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Whether to attempt to enable TF eager execution.")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Whether to attempt to enable tracing for eager mode.")
    parser.add_argument(
        "--log-flatland-stats",
        action="store_true",
        default=True,
        help="Whether to log additional flatland specfic metrics such as percentage complete or normalized score.")
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Whether to run evaluation. Default evaluation config is default.yaml "
        "to use custom evaluation config set (eval_generator:high_eval) under configs")
    parser.add_argument(
        "--bind-all",
        action="store_true",
        default=False,
        help="Whether to expose on network (binding on all network interfaces).")
    parser.add_argument(
        "--env", default=None, type=str, help="The gym environment to use.")
    parser.add_argument(
        "--queue-trials",
        action="store_true",
        help=(
            "Whether to queue trials when the cluster does not currently have "
            "enough resources to launch one. This should be set to True when "
            "running on an autoscaling cluster to enable automatic scale-up."))
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
             "overrides any trial-specific options set via flags above.")
    return parser
