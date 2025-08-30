# Standard library imports
from pathlib import Path
from sys import exit
from typing import Optional, Self

# Third-party imports
import pandas as pd
import yaml
from library import Logger
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict


class ConfigData(BaseModel):
    """
    Configuration data model defining project-related file paths and settings.

    Attributes:
        root (Path): The root directory of the project, typically two levels up from this file.
        project_name (str): Name of the project, used in naming files.
        training_file (Path): Path to the training data file (e.g., 'path_name/train.csv').
        target_name (Optional[str]): Name of the target column, if none and no competition
            data is available, last column will be selected
        competition_file (Optional[Path]): Optional path to the competition data file (e.g., 'path_name/test.csv').
        submission_file (Path): Path to the project submission file (e.g., 'path_name/project_submission.csv').
        report_template (Path): Path to the project report template file (e.g., 'path_name/project_report.html').
        log_file (Path): Path to the project log file (e.g., 'path_name/project.log').

    Config:
        json_encoders (dict): Custom JSON serialization for fields; encodes Path objects as strings.

    Methods:
        save_to_yaml(filename: str) -> None:
            Serialize the config data to a YAML file.

        load_from_yaml(filename: str) -> ConfigData:
            Load the config data from a YAML file and return a ConfigData instance.
    """

    root: Path = Path(__file__).parent.parent
    project_name: str  # name of the project, also used in filenames
    training_file: Path  # e.g. path_name/train.csv
    target: Optional[str]  # target column
    competition_file: Optional[Path]  # e.g. path_name/test.csv
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
    def load_from_yaml(cls, filename: str) -> Self:
        with open(file=filename, mode="r", encoding="utf-8") as file:
            data = yaml.safe_load(stream=file)
        return cls(**data)


class OriginalData(BaseModel):
    """
    Data model encapsulating the original datasets used for training and competition.

    Attributes:
        X_train (DataFrame): The training features dataset used for model training.
        y_train (DataFrame): The target dataset corresponding to X_train, containing labels or outcomes.
        X_comp (Optional[DataFrame]): Optional competition dataset for evaluation or prediction purposes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    X_train: DataFrame  # training data
    y_train: pd.Series  # target data
    X_comp: Optional[DataFrame]  # competition data


class ModellingData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    X_train: DataFrame  # training data
    y_train: DataFrame  # target data
    X_val: DataFrame  # training data
    y_val: DataFrame  # target data
    X_test: DataFrame  # training data
    y_test: DataFrame  # target data


def read_data(file_path: Path, logger) -> Optional[pd.DataFrame]:
    logger.info(f"[BLUE]- Reading data from {file_path}")
    if not file_path:
        return None
    try:
        if file_path.suffix == ".xlsx":
            return pd.read_excel(io=file_path)
        else:
            return pd.read_csv(filepath_or_buffer=file_path)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except PermissionError:
        logger.error(f"File is still open: {file_path}")
        input("Close the file and press [ENTER]")
        read_data(file_path=file_path, logger=logger)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


def load_data(config: ConfigData, logger: Logger) -> OriginalData:

    X_train_full = read_data(file_path=config.training_file, logger=logger)
    if X_train_full is None:
        logger.error(
            msg=f"Training data could not be loaded from {config.training_file}"
        )
        exit("Exiting due to missing training data")

    # Drop target column to
    if config.competition_file is not None:
        X_comp = read_data(file_path=config.competition_file, logger=logger)
    else:
        X_comp = None
    if config.target is None or config.target != "":
        if X_comp is not None:
            target = list(set(X_train_full.columns) - set(X_comp.columns))
            # Assuming exactly one target column is missing, get it as a string
            if len(target) == 1:
                config.target = target[0]
            else:
                raise ValueError(
                    "Expected exactly one target column missing in X_comp"
                )
        else:
            config.target = X_train_full.columns[-1]
    y_train = X_train_full[config.target]  # Extract target column
    X_train = X_train_full.drop(columns=[config.target])
    return OriginalData(X_train=X_train, y_train=y_train, X_comp=X_comp)
