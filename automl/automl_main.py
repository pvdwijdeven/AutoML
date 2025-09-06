# Standard library imports
import argparse
from pathlib import Path

# Local application imports
from automl.dataloader import ConfigData, OriginalData, get_config, load_data
from automl.eda import perform_eda
from automl.library import Logger


def get_args() -> tuple[str, Path]:
    parser = argparse.ArgumentParser(
        description="Automated Machine Learning application"
    )
    parser.add_argument(
        "--title", type=str, required=False, help="Title of the project"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        help="Path to config file (.yaml)",
    )

    args = parser.parse_args()
    return args.title, args.config


def main() -> None:
    """
    Main entry point of the application.
    Loads configuration, initializes logging, loads data, and prints dataset shapes.
    """
    title, config = get_args()
    config_data: ConfigData = get_config(title=title, config=config)

    logger = Logger(
        level_console=Logger.INFO,
        level_file=Logger.DEBUG,
        filename=config_data.log_file,
        wx_handler=None,
    )

    original_data: OriginalData = load_data(
        config_data=config_data, logger=logger
    )
    perform_eda(config_data=config_data, original_data=original_data)


if __name__ == "__main__":
    main()
