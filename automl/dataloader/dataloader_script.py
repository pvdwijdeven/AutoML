# Standard library imports
import importlib
import importlib.util
import os
import sys
from dataclasses import fields, dataclass, field
from pathlib import Path
from sys import exit
from typing import Optional

# Third-party imports
import pandas as pd
import yaml
from pandas import DataFrame

# Local application imports
from automl.library import Logger


@dataclass
class ConfigData:
    """
    Configuration data model for project paths and settings.

    This model defines file paths and project-specific settings
    for training, evaluation, reporting, and logging.
    """

    root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )
    project_name: str = ""
    training_file: Path = Path()
    submission_file: Path = Path()
    report_template: Path = Path()
    target: Optional[str] = None
    competition_file: Optional[Path] = None
    config_file: Optional[Path] = None
    description_file: Optional[Path] = None
    update_file: Optional[Path] = None
    log_file: Path = Path()


    def __iter__(self):
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def save_to_yaml(self) -> None:
        """
        Saves the configuration object to a YAML file.

        Raises FileNotFoundError if 'config_file' is None or invalid.
        """
        if not self.config_file:
            raise FileNotFoundError("config_file attribute is not set.")

        parent_dir = self.config_file.parent
        os.makedirs(parent_dir, exist_ok=True)

        # Convert dataclass to dictionary, converting Paths to strings for YAML
        def serialize(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, (list, tuple)):
                return [serialize(o) for o in obj]
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj

        data_dict = {k: serialize(v) for k, v in self.__dict__.items()}

        with open(self.config_file, "w", encoding="utf-8") as file:
            yaml.safe_dump(data_dict, file, sort_keys=False)

    @classmethod
    def load_from_yaml(cls, filename: Path) -> "ConfigData":
        """
        Loads a configuration object from a YAML file.

        Raises FileNotFoundError, yaml.YAMLError on error.
        """
        with open(filename, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        # Convert string paths back to Path objects where relevant
        path_fields = {
            "root",
            "training_file",
            "submission_file",
            "report_template",
            "competition_file",
            "config_file",
            "description_file",
            "update_file",
            "log_file",
        }

        for field_name in path_fields:
            if field_name in data and data[field_name] is not None:
                data[field_name] = Path(data[field_name])

        return cls(**data)


@dataclass
class OriginalData():
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

    X_train: DataFrame
    y_train: pd.Series
    X_comp: Optional[DataFrame]

@dataclass
class ModellingData():
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
        update_file=None,
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
    original_data = OriginalData(
        X_train=X_train, y_train=y_train, X_comp=X_comp
    )
    if config_data.update_file is not None:
        original_data = update_with_user_function(
            config_data=config_data, original_data=original_data, logger=logger
        )
    return original_data


def update_with_user_function(
    config_data: ConfigData, original_data: OriginalData, logger: Logger
) -> OriginalData:

    if config_data.update_file is not None:
        module_name = config_data.update_file.stem  # e.g. "user_file"

        # Load module from file
        spec = importlib.util.spec_from_file_location(
            module_name, str(config_data.update_file)
        )
        assert spec is not None
        assert spec.loader is not None
        user_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = user_module
        spec.loader.exec_module(user_module)

        # Get the function
        update_func = getattr(user_module, "update_df", None)
        if update_func is None:
            logger.error(
                msg=f"No function 'update_df' found in {config_data.update_file}, skipping update"
            )
        else:
            original_data.X_train = update_types(
                X_train=update_func(original_data.X_train)
            )
            if original_data.X_comp is not None:
                original_data.X_comp = update_types(
                    X_train=update_func(original_data.X_comp)
                )
    # Apply the function to the dataframe
    return original_data


def update_types(X_train: DataFrame):
    X_work = X_train.copy()
    for col in X_work.columns:
        series = X_work[col]

        # Skip non-object/string types
        if pd.api.types.is_numeric_dtype(arr_or_dtype=series):
            continue

        # Try integer
        as_int = pd.to_numeric(arg=series, errors="coerce")
        if as_int.notna().all():
            if (as_int % 1 == 0).all():
                X_work[col] = as_int.astype(dtype="Int64")  # nullable integer
                continue
            else:
                X_work[col] = as_int.astype(dtype=float)
                continue

        # Try float
        if as_int.notna().sum() > 0:
            X_work[col] = as_int.astype(dtype=float)
            continue
    return X_work
