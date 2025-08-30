# Standard library imports
import argparse
import os
from pathlib import Path


from dataloader import OriginalData, load_data, ConfigData
from library import Logger


def ensure_folder_exists(root, filename) -> None:
    if filename == "" or filename is None:
        return
    # Extract folder path from filename relative to root
    folder_path = os.path.dirname(os.path.join(root, filename))
    # Check if folder exists, and create recursively if not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


def create_config_paths(config: ConfigData) -> None:
    for field_name, value in config:
        if field_name not in ["root", "project_name"]:
            ensure_folder_exists(root=config.root, filename=value)


def get_config_from_title(title) -> ConfigData:
    config = ConfigData(
        project_name=title,
        training_file=Path(f"personal/{title}/data/train.csv"),
        target=None,
        competition_file=Path(f"personal/{title}/data/test.csv"),
        submission_file=Path(f"personal/{title}/data/submission.csv"),
        report_template=Path(f"personal/{title}/export/{title}_report.html"),
        log_file=Path(f"personal/{title}/log/{title}.log"),
    )
    filename = f"personal/yaml/{config.project_name}.yaml"
    ensure_folder_exists(root=config.root, filename=filename)
    config.save_to_yaml(filename=filename)
    return config


def get_config() -> ConfigData:
    parser = argparse.ArgumentParser(
        description="Automated Machine Learning application"
    )
    # Add argument for title (string)
    parser.add_argument(
        "--title", type=str, required=False, help="Title of the project"
    )
    # Add argument for config file path (string)
    parser.add_argument(
        "--config", type=str, required=False, help="Path to config file (.yaml)"
    )
    args = parser.parse_args()

    if args.config:
        config = ConfigData.load_from_yaml(filename="config.yaml")
    else:
        if args.title:
            title = args.title
        else:
            title = "Titanic"
        config = get_config_from_title(title=title)
    create_config_paths(config=config)
    return config


def main() -> None:
    config: ConfigData = get_config()
    logger = Logger(
        level_console=Logger.INFO,
        level_file=Logger.DEBUG,
        filename=config.log_file,
        wx_handler=None,
    )
    original_data: OriginalData = load_data(config=config, logger=logger)
    print(original_data.X_train.shape)
    print(original_data.y_train.shape)
    if original_data.X_comp is not None:
        print(original_data.X_comp.shape)


if __name__ == "__main__":
    main()
