# Standard library imports
import os
from pathlib import Path
from sys import exit
from typing import Optional, Self

# Third-party imports
import pandas as pd
import yaml
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict

# Local application imports
from automl.library import Logger


class ConfigData(BaseModel):
    """
    Configuration data model for project paths and settings.

    This model defines file paths and project-specific settings
    for training, evaluation, reporting, and logging.
    All path attributes are automatically serialized as strings.

    Attributes
    ----------
    root : pathlib.Path
        Root directory of the project, typically two levels up from this file.
    project_name : str
        Project name, used for directory and file naming conventions.
    training_file : pathlib.Path
        Path to the training dataset file (CSV or Excel).
    target : Optional[str]
        Target column name in the training dataset. If None and no
        competition file is given, uses the last column in the training data.
    competition_file : Optional[pathlib.Path]
        Optional path to the competition/test dataset.
    submission_file : pathlib.Path
        Path for saving generated submission files.
    report_template : pathlib.Path
        Path to the report template file for the project.
    config_file : Optional[pathlib.Path]
        Path to current configuration YAML file (optional).
    description_file: Optional[Path]
        Path to column description YAML file (optiomal)
    log_file : pathlib.Path
        Path for the main project logging file.

    Methods
    -------
    save_to_yaml() -> None
        Saves the current configuration to the YAML file specified by config_file.
    load_from_yaml(filename: pathlib.Path) -> ConfigData
        Loads and returns a configuration object from the provided YAML file.

    Notes
    -----
    All path-type fields are serialized as strings in JSON/YAML output.
    """

    root: Path = Path(__file__).parent.parent.parent
    project_name: str
    training_file: Path
    submission_file: Path
    report_template: Path
    target: Optional[str]
    competition_file: Optional[Path]
    config_file: Optional[Path]
    description_file: Optional[Path]
    log_file: Path

    class Config:
        # Force serialization of Path objects as strings
        json_encoders = {Path: str}

    def save_to_yaml(self) -> None:
        """
        Saves the configuration object to a YAML file.

        The file path is taken from the 'config_file' attribute.
        Ensures that the parent folder exists before writing.

        Raises
        ------
        FileNotFoundError
            If the 'config_file' attribute is not set or invalid.
        """
        ensure_folder_exists(root=self.root, filename=self.config_file)
        with open(
            file=str(object=self.config_file), mode="w", encoding="utf-8"
        ) as file:
            yaml.safe_dump(
                data=self.model_dump(mode="json"),
                stream=file,
                sort_keys=False,
            )

    @classmethod
    def load_from_yaml(cls, filename: Path) -> Self:
        """
        Loads a configuration object from a YAML file.

        Parameters
        ----------
        filename : pathlib.Path
            Path to the YAML configuration file.

        Returns
        -------
        ConfigData
            Instantiated configuration object with settings loaded from file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at the provided path.
        yaml.YAMLError
            If the file is not a valid YAML format.
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


def ensure_folder_exists(root: Path, filename: str | Path | None) -> None:
    """
    Ensure that the folder required for a given file inside `root` exists.
    If the folder does not exist, it is created recursively.

    Parameters
    ----------
    root : Path
        The root directory against which the filename is resolved.
    filename : str | Path | None
        The relative filename within `root`. If None or empty, nothing is done.
    """
    if not filename:
        return

    folder_path = os.path.dirname(os.path.join(str(root), str(filename)))
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


def create_config_paths(config: ConfigData) -> None:
    """
    Create all necessary directories for the paths specified in a configuration.

    Parameters
    ----------
    config : ConfigData
        The configuration object containing file and directory paths.
    """
    for field_name, value in config:
        if field_name not in ("root", "project_name"):
            ensure_folder_exists(root=config.root, filename=value)


def get_config_from_title(title: str) -> ConfigData:
    """
    Create a `ConfigData` object from a given project title.
    Automatically generates standard paths for training, testing,
    logging, and reporting.

    Parameters
    ----------
    title : str
        The title or project name.

    Returns
    -------
    ConfigData
        The generated configuration object, also saved as YAML.
    """
    config = ConfigData(
        project_name=title,
        training_file=Path(f"personal/{title}/data/train.csv"),
        target=None,
        competition_file=Path(f"personal/{title}/data/test.csv"),
        submission_file=Path(f"personal/{title}/data/submission.csv"),
        report_template=Path(f"personal/{title}/export/{title}_report.html"),
        config_file=Path(f"personal/yaml/{title}.yaml"),
        description_file=None,
        log_file=Path(f"personal/{title}/log/{title}.log"),
    )
    config.save_to_yaml()
    return config


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
        logger.error(msg=f"File is open and cannot be read: {file_path}")
        input("Please close the file and press [ENTER] to retry...")
        return read_data(file_path=file_path, logger=logger)
    except Exception as e:
        logger.error(msg=f"Error reading {file_path}: {e}")
        return None


def get_config(title, config) -> ConfigData:
    """
    Parse command-line arguments to load or generate a configuration.

    Returns
    -------
    ConfigData
        The configuration object loaded from file or generated by title.
    """

    if config:
        config = ConfigData.load_from_yaml(filename=config)
    else:
        title = title if title else "Titanic"
        config = get_config_from_title(title=title)

    create_config_paths(config=config)
    return config


def load_data(config_data: ConfigData, logger: Logger) -> OriginalData:
    """
    Load training and optional competition data from configuration.
    Automatically extracts features and target column.

    Parameters
    ----------
    config_data : ConfigData
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
    X_train_full = read_data(file_path=config_data.training_file, logger=logger)
    if X_train_full is None:
        logger.error(
            msg=f"Training data could not be loaded from {config_data.training_file}"
        )
        exit("Exiting due to missing training data")

    # Load competition dataset if specified
    X_comp = None
    if config_data.competition_file:
        X_comp = read_data(
            file_path=config_data.competition_file, logger=logger
        )

    # Resolve target column if missing
    if not config_data.target:
        if X_comp is not None:
            # Target is the column not present in the competition set
            target_candidates = list(
                set(X_train_full.columns) - set(X_comp.columns)
            )
            if len(target_candidates) == 1:
                config_data.target = target_candidates[0]
            else:
                raise ValueError(
                    f"Expected exactly one target column missing in competition "
                    f"data, found: {target_candidates}"
                )
        else:
            # Default to last column in training dataset
            config_data.target = X_train_full.columns[-1]
        # update yaml file
        if config_data.config_file is not None:
            config_data.save_to_yaml()
    # Split into features and target
    y_train = X_train_full[config_data.target]
    X_train = X_train_full.drop(columns=[config_data.target])

    return OriginalData(X_train=X_train, y_train=y_train, X_comp=X_comp)
