import argparse
import yaml
from solver import Solver

def get_parser():
    parser = argparse.ArgumentParser(description="sEMG-based gesture recognition")
    parser.add_argument(
        "--config",
        "-cfg",
        default="./config/bandmyo.yaml",
        type=str,
        help="Config file which is used.",
    )
    parser.add_argument("--stage", "-sg", type=str, choices=["pretrain", "train", "test"], default="train", help="train stage or test stage")

    # yaml args
    parser.add_argument("--subjects", "-s", nargs="*", type=int, default=None)
    parser.add_argument("--num_epochs", "-ne", type=int, default=None)
    parser.add_argument("--batch_size", "-bs", type=int, default=None)
    parser.add_argument("--window_size", "-wz", type=int, default=None)
    parser.add_argument("--window_step", "-ws", type=int, default=None)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        # default_arg = yaml.load(f, Loader=yaml.FullLoader)
        default_arg = yaml.load(f)

    # update args if specified on the command line
    args = vars(args)
    keys = list(args.keys())
    for key in keys:
        if args[key] is None:
            del args[key]
    default_arg.update(args)
    parser.set_defaults(**default_arg)

    args = parser.parse_args()
    # for k, v in vars(args).items():
    #     print(k, ": ", v)
    # print(yaml.dump(vars(args)))
    solver = Solver(args)
    solver.start(task="intra_session")
