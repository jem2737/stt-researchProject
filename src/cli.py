import argparse
from listen import listen
from train import train

def main():
    parser = argparse.ArgumentParser(description="This is a tool to classify audio into emotional classes. The tool can also listen and classify live audio")
    subparsers = parser.add_subparsers(dest="command")

    # train command
    train_parser = subparsers.add_parser("train", help="Train classifier")
    train_parser.add_argument("--raw-data-path", type=str, default=None)
    train_parser.add_argument("--clean-data-path", type=str, default=None)
    train_parser.add_argument("--classifier-path", type=str, default=None)
    train_parser.add_argument("--clean-data-json", type=str, default=None)
    train_parser.add_argument("--prepare-data", action="store_true")
    train_parser.add_argument("--no-prepare-data", action="store_true")
    # listen command
    listen_parser = subparsers.add_parser("listen", help="Run live listening")
    listen_parser.add_argument("--classifier-path", type=str, default=None)

    args = parser.parse_args()

    if args.command == "train":
        prepare_data_bool = True
        if args.no_prepare_data:
            prepare_data_bool = False
        elif args.prepare_data:
            prepare_data_bool = True
        train(
            clean_data_json=args.clean_data_json,
            classifier_path=args.classifier_path,
            raw_data_path=args.raw_data_path,
            clean_data_path=args.clean_data_path,
            prepare_data=prepare_data_bool
        )

    elif args.command == "listen":
        listen(classifier_path=args.classifier_path)

if __name__ == "__main__":
    main()