# Standard library imports
from pathlib import Path
from sys import exit
from typing import Optional, Self

# Third-party imports
import pandas as pd
import yaml
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict

# Local imports
from automl.library import Logger


class ConfigData(BaseModel):
    """
    Configuration data model defining project-related file paths and settings.

    Attributes
    ----------
    root : Path
        Root directory of the project, typically two levels up from this file.
    project_name : str
        Project name, also used in file naming conventions.
    training_file : Path
        Path to the training dataset (CSV or Excel).
    target : Optional[str]
        Target column in the training dataset.
        If None and no competition file is provided, the last column will be used.
    competition_file : Optional[Path]
        Path to competition/test dataset (optional).
    submission_file : Path
        Path where submission results will be saved.
    report_template : Path
        Path to the project report file.
    log_file : Path
        Path for logging file.

    Methods
    -------
    save_to_yaml(filename: str) -> None
        Save this configuration to a YAML file.
    load_from_yaml(filename: str) -> ConfigData
        Load configuration from a YAML file.
    """

    root: Path = Path(__file__).parent.parent.parent
    project_name: str
    training_file: Path
    target: Optional[str]
    competition_file: Optional[Path]
    submission_file: Path
    report_template: Path
    log_file: Path

    class Config:
        # Force serialization of Path objects as strings
        json_encoders = {Path: str}

    def save_to_yaml(self, filename: str) -> None:
        """
        Save this configuration to a YAML file.

        Parameters
        ----------
        filename : str
            Path where the YAML file should be saved.
        """
        with open(file=filename, mode="w", encoding="utf-8") as file:
            yaml.safe_dump(
                data=self.model_dump(mode="json"),
                stream=file,
                sort_keys=False,
            )

    @classmethod
    def load_from_yaml(cls, filename: str) -> Self:
        """
        Load a configuration object from a YAML file.

        Parameters
        ----------
        filename : str
            Path to the YAML configuration file.

        Returns
        -------
        ConfigData
            A configuration object loaded from the file.
        """
        with open(file=filename, mode="r", encoding="utf-8") as file:
            data = yaml.safe_load(stream=file)
        return cls(**data)


class OriginalData(BaseModel):
    """
    Container for original datasets used for training and competition.

    Attributes
    ----------
    X_train : DataFrame
        Training features (all columns except target).
    y_train : pd.Series
        Target variable from the training dataset.
    X_comp : Optional[DataFrame]
        Competition dataset (e.g., test features), if provided.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X_train: DataFrame
    y_train: pd.Series
    X_comp: Optional[DataFrame]


class ModellingData(BaseModel):
    """
    Data model for holding training, validation, and test datasets.

    Attributes
    ----------
    X_train : DataFrame
        Training features.
    y_train : DataFrame
        Training target values.
    X_val : DataFrame
        Validation features.
    y_val : DataFrame
        Validation target values.
    X_test : DataFrame
        Test features.
    y_test : DataFrame
        Test target values.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X_train: DataFrame
    y_train: DataFrame
    X_val: DataFrame
    y_val: DataFrame
    X_test: DataFrame
    y_test: DataFrame


def read_data(file_path: Path, logger: Logger) -> Optional[pd.DataFrame]:
    """
    Read data from CSV or Excel file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the data file.
    logger : Logger
        Logger object for recording process and errors.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame if file is loaded successfully, None otherwise.
    """
    if not file_path:
        return None

    logger.info(f"[BLUE]- Reading data from {file_path}")

    try:
        if file_path.suffix == ".xlsx":
            return pd.read_excel(io=file_path)
        return pd.read_csv(filepath_or_buffer=file_path)

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except PermissionError:
        logger.error(f"File is open and cannot be read: {file_path}")
        input("Please close the file and press [ENTER] to retry...")
        return read_data(file_path=file_path, logger=logger)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


def load_data(config: ConfigData, logger: Logger) -> OriginalData:
    """
    Load training and optional competition data from configuration.
    Automatically extracts features and target column.

    Parameters
    ----------
    config : ConfigData
        Configuration containing paths to datasets and target info.
    logger : Logger
        Logger for recording info and errors.

    Returns
    -------
    OriginalData
        Container with training features, target, and competition data.

    Raises
    ------
    ValueError
        If more than one potential target column is found when deducing target.
    SystemExit
        If training data cannot be loaded.
    """
    # Load training dataset
    X_train_full = read_data(file_path=config.training_file, logger=logger)
    if X_train_full is None:
        logger.error(
            f"Training data could not be loaded from {config.training_file}"
        )
        exit("Exiting due to missing training data")

    # Load competition dataset if specified
    X_comp = None
    if config.competition_file:
        X_comp = read_data(file_path=config.competition_file, logger=logger)

    # Resolve target column if missing
    if not config.target:
        if X_comp is not None:
            # Target is the column not present in the competition set
            target_candidates = list(
                set(X_train_full.columns) - set(X_comp.columns)
            )
            if len(target_candidates) == 1:
                config.target = target_candidates[0]
            else:
                raise ValueError(
                    f"Expected exactly one target column missing in competition "
                    f"data, found: {target_candidates}"
                )
        else:
            # Default to last column in training dataset
            config.target = X_train_full.columns[-1]

    # Split into features and target
    y_train = X_train_full[config.target]
    X_train = X_train_full.drop(columns=[config.target])

    return OriginalData(X_train=X_train, y_train=y_train, X_comp=X_comp)
