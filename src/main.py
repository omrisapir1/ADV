import argparse
from .train import run


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True, help='Path to YAML config file.')
    return ap.parse_args()


def main():
    args = parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
