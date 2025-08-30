# Standard library imports
import argparse
import os
from pathlib import Path

# Third-party imports
import yaml
from pydantic import BaseModel

# from devtools import debug


class Config_Data(BaseModel):
    root: Path = Path(__file__).parent.parent
    project_name: str  # name of the project, also used in filenames
    training_file: Path  # e.g. path_name/train.csv
    test_file: Path  # e.g. path_name/test.csv
    submission_file: Path  # e.g. path_name/project_submission.csv
    report_template: Path  # e.g. path_name/project_report.html
    log_file: Path  # e.g. path_name/project.log

    class Config:
        # force serialization of Path as str
        json_encoders = {Path: str}

    def save_to_yaml(self, filename: str) -> None:
        with open(file=filename, mode="w", encoding="utf-8") as file:
            yaml.safe_dump(
                data=self.model_dump(mode="json"),
                stream=file,
                sort_keys=False,
            )

    @classmethod
    def load_from_yaml(cls, filename: str):
        with open(file=filename, mode="r", encoding="utf-8") as file:
            data = yaml.safe_load(stream=file)
        return cls(**data)


def ensure_folder_exists(root, filename) -> None:
    # Extract folder path from filename relative to root
    folder_path = os.path.dirname(os.path.join(root, filename))
    # Check if folder exists, and create recursively if not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


def create_config_paths(config: Config_Data) -> None:
    for field_name, value in config:
        if field_name not in ["root", "project_name"]:
            ensure_folder_exists(root=config.root, filename=value)


def get_config_from_title(title) -> Config_Data:
    config = Config_Data(
        project_name=title,
        training_file=Path(f"personal/{title}/data/train.csv"),
        test_file=Path(f"personal/{title}/data/test.csv"),
        submission_file=Path(f"personal/{title}/data/submission.csv"),
        report_template=Path(f"personal/{title}/export/{title}_report.html"),
        log_file=Path(f"personal/{title}/log/{title}.log"),
    )
    filename = f"personal/yaml/{config.project_name}.yaml"
    ensure_folder_exists(root=config.root, filename=filename)
    config.save_to_yaml(filename=filename)
    return config


def main() -> None:
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
        config = Config_Data.load_from_yaml(filename="config.yaml")
    else:
        if args.title:
            title = args.title
        else:
            title = "Titanic"
        config = get_config_from_title(title=title)
    create_config_paths(config=config)


if __name__ == "__main__":
    main()
