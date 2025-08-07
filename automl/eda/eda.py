import pandas as pd
from jinja2 import Environment, FileSystemLoader
from .overview import create_overview_table
from .missing import (
    missing_data_summary,
    plot_missingness_matrix,
    plot_missing_correlation,
    generate_missing_summary,
)
from .testdata import analyze_test_data
from scipy.stats import skew
from library import (
    infer_dtype,
    Logger,
    get_html_from_template,
    get_frequency_table,
    analyze_string_column,
    analyze_categorical_column,
    analyze_numeric_column,
    analyze_boolean_column,
    analyze_target,
    generate_feature_relations,
    generate_relation_visuals,
    generate_eda_plots,
)

import numpy as np
from scipy.stats import entropy as scipy_entropy
from datetime import datetime
import importlib.util
import sys
import pathlib


class AutoML_EDA:
    """
    AutoML_EDA is a class for automated exploratory data analysis (EDA) on tabular datasets.
    It generates a comprehensive HTML report summarizing dataset features, relations, missing values,
    and recommendations for further analysis or modeling.
    Attributes:
        report_file (str): Path to save the generated EDA report.
        file_train (str): Path to the training dataset file (CSV or XLSX).
        file_test (str): Path to the test dataset file (optional).
        title (str): Title for the EDA report.
        target (str): Name of the target column for analysis.
        description (str): Path to a file containing column descriptions (optional).
        nogui (bool): If True, disables GUI-related logging.
        update_script (str): Path to a user-defined script for custom DataFrame updates (optional).
        logger (Logger): Logger instance for logging messages.
    Methods:
        parse_column_descriptions_to_html(text):
            Parses column descriptions from text and formats them as HTML.
        read_description(file_path):
            Reads and parses column descriptions from a file.
        read_data(file_path) -> pd.DataFrame | None:
            Loads data from a CSV or XLSX file into a pandas DataFrame.
        update_with_user_function(df, filepath):
            Applies a user-defined function from a script to update the DataFrame.
        column_type(column_name: str) -> None:
            Infers and converts the data type of a column, synchronizing train and test sets.
        analyse_column(column_name: str):
            Analyzes a single column, returning statistics and suggestions based on its type.
        analyse_columns() -> None:
            Performs type inference and analysis for all columns in the training set.
        perform_eda() -> str:
            Executes the full EDA workflow, generates the HTML report, and saves it to file.
    Usage:
        Instantiate the class with dataset paths and parameters, then call `perform_eda()` to generate the report.
    Example:
        eda = AutoML_EDA(
            report_file="eda_report.html",
            file_train="train.csv",
            file_test="test.csv",
            title="My Dataset",
            target="target_column",
            description="column_descriptions.txt"
        eda.perform_eda()
    """

    def __init__(
        self,
        report_file: str,
        file_train: str,
        file_test: str = "",
        title: str = "",
        target: str = "",
        description: str = "",
        nogui=True,
        update_script: str = "",
        logger: Logger | None = None,
    ) -> None:

        self.report_file = report_file
        self.file_train = file_train
        self.file_test = file_test
        self.title = title
        self.target = target
        self.description = description
        self.nogui = nogui
        self.update_script = update_script
        if logger is None:
            self.logger = Logger(
                level_console=Logger.INFO,
                level_file=Logger.DEBUG,
                filename="",
                wx_handler=None,
            )
        else:
            self.logger = logger

    def parse_column_descriptions_to_html(self, text):
        result = {}
        current_column = None
        current_description = []

        for line in text:
            line = line.strip()
            if not line:  # Empty line
                # If we have a current column, save it to the dictionary
                if current_column is not None:
                    # Join with HTML line breaks and strip whitespace
                    description = "<br>".join(current_description).strip()
                    result[current_column] = description
                    current_column = None
                    current_description = []
                continue

            # If we're not currently processing a column, this must be a new column name
            if current_column is None:
                current_column = line
            else:
                current_description.append(
                    line.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
                )

        # Add the last column if there wasn't an empty line after it
        if current_column is not None:
            description = "<br>".join(current_description).strip()
            result[current_column] = description

        return result

    def read_description(self, file_path):
        self.logger.info(
            f"[BLUE]- Reading column descriptions from {file_path}"
        )
        if not file_path:
            return {}
        try:
            with open(file_path, "r") as f:
                description = f.readlines()
            return self.parse_column_descriptions_to_html(description)
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return {}

    def read_data(self, file_path) -> pd.DataFrame | None:
        self.logger.info(f"[BLUE]- Reading data from {file_path}")
        if not file_path:
            return None
        try:
            if file_path.endswith(".xlsx"):
                return pd.read_excel(file_path)
            else:
                return pd.read_csv(file_path)
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return None

    def update_with_user_function(self, df, filepath):

        # Ensure absolute path and module name
        filepath = pathlib.Path(filepath).resolve()
        module_name = filepath.stem  # e.g. "user_file"

        # Load module from file
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        assert spec is not None
        assert spec.loader is not None
        user_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = user_module
        spec.loader.exec_module(user_module)

        # Get the function
        update_func = getattr(user_module, "update_df", None)
        if update_func is None:
            raise AttributeError(f"No function 'update_df' found in {filepath}")

        # Apply the function to the dataframe
        return update_func(df)

    def column_type(
        self,
        column_name: str,
    ) -> None:

        def convert_to_bool_if_binary(series):
            if pd.api.types.is_numeric_dtype(
                series
            ) and not pd.api.types.is_bool_dtype(series):
                unique_vals = set(series.dropna().unique())
                if unique_vals <= {0, 1}:
                    return series.map({0: False, 1: True}).astype("boolean")
            return series

        def normalize_booleans(series: pd.Series) -> pd.Series:

            if pd.api.types.is_object_dtype(
                series
            ) or pd.api.types.is_string_dtype(series):

                normalized = series.dropna().map(
                    lambda x: str(x).strip().lower()
                )
                valid_values = {"true", "false"}

                unexpected = normalized[~normalized.isin(valid_values)]
                if not unexpected.empty:
                    return series

                return series.map(
                    lambda x: (
                        True
                        if str(x).strip().lower() == "true"
                        else (
                            False
                            if str(x).strip().lower() == "false"
                            else pd.NA
                        )
                    )
                ).astype("boolean")
            return series

        def convert_to_numeric(series: pd.Series) -> pd.Series:
            return pd.to_numeric(
                series.astype(str)
                .str.strip()
                .str.replace(r'^[\'"]+|[\'"]+$', "", regex=True),
                errors="coerce",
            )

        assert (
            self.df_train is not None
        ), "Training DataFrame is not initialized"
        col_series_train = self.df_train[column_name]

        # Step 1: Normalize booleans
        col_series_train = convert_to_bool_if_binary(col_series_train)

        col_series_train = normalize_booleans(col_series_train)

        # Step 2: Detect if boolean
        unique_values = pd.Series(col_series_train.dropna().unique())
        if unique_values.isin([True, False]).all():
            self.df_train[column_name] = col_series_train.astype("boolean")

            if self.df_test is not None and column_name in self.df_test.columns:
                self.df_test[column_name] = normalize_booleans(
                    self.df_test[column_name]
                ).astype("boolean")
        # Step 3: Categorical vs String
        inferred_dtype = None
        if pd.api.types.is_object_dtype(self.df_train[column_name]):
            unique_ratio = self.df_train[column_name].nunique(
                dropna=True
            ) / len(self.df_train)
            inferred_dtype = "category" if unique_ratio < 0.1 else "string"
            self.df_train[column_name] = self.df_train[column_name].astype(
                inferred_dtype
            )

        # Step 4: Attempt numeric conversion
        try_numeric = pd.api.types.is_object_dtype(
            col_series_train
        ) or pd.api.types.is_string_dtype(col_series_train)

        if try_numeric:
            try:
                converted = convert_to_numeric(col_series_train)
            except Exception as e:
                self.logger.warning(
                    f"Conversion to numeric failed for column '{column_name}': {e}"
                )
                converted = None
        elif pd.api.types.is_numeric_dtype(
            col_series_train
        ) and not pd.api.types.is_bool_dtype(col_series_train):
            converted = col_series_train.copy()
        else:
            converted = None

        if converted is not None:
            is_int_like = np.isclose(converted.dropna() % 1, 0).all()
            non_na_count = col_series_train.notna().sum()
            if converted.notna().sum() == non_na_count:
                if pd.api.types.is_float_dtype(converted) and is_int_like:
                    self.df_train[column_name] = converted.astype("Int64")
                    inferred_dtype = "Int64"
                elif is_int_like:
                    self.df_train[column_name] = converted.astype("Int64")
                    inferred_dtype = "Int64"
                else:
                    self.df_train[column_name] = converted
                    inferred_dtype = str(converted.dtype).capitalize()

        # Step 5: Apply same dtype to df_test
        if self.df_test is not None and column_name in self.df_test.columns:
            col_test = self.df_test[column_name]
            if inferred_dtype == "boolean":
                self.df_test[column_name] = normalize_booleans(col_test).astype(
                    "boolean"
                )
            elif inferred_dtype == "category":
                self.df_test[column_name] = col_test.astype("category")
            elif inferred_dtype == "string":
                self.df_test[column_name] = col_test.astype("string")
            elif inferred_dtype == "Int64":
                try:
                    converted_test = convert_to_numeric(col_test)
                    self.df_test[column_name] = converted_test.astype("Int64")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to apply Int64 conversion to test column '{column_name}': {e}"
                    )
            elif inferred_dtype is not None and inferred_dtype.startswith(
                "Float"
            ):
                try:
                    converted_test = convert_to_numeric(col_test)
                    # Use numpy dtype object for astype

                    self.df_test[column_name] = converted_test.astype(
                        np.float64
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to apply float conversion to test column '{column_name}': {e}"
                    )
        self.type_conversion[column_name]["actual"] = self.df_train[
            column_name
        ].dtype

    def analyse_column(self, column_name: str):

        assert (
            self.df_train is not None
        ), "Training DataFrame is not initialized"

        # type boolean:
        # - missing values, frequency True/False, graph
        if self.type_conversion[column_name]["new"] == "boolean":
            frequency = get_frequency_table(self.df_train, column_name)
            missing_count = self.df_train[column_name].isna().sum()
            missing_pct = missing_count / len(self.df_train) * 100
            return {
                "type": f"boolean ({self.type_conversion[column_name]['actual']}) - was {self.type_conversion[column_name]['original']}",
                "description": self.dict_description.get(column_name, ""),
                "missing_values": f"{missing_count} ({missing_pct:.1f}%)",
                "frequency": frequency.replace("np.", "").replace("_", ""),
                "suggestions": "- "
                + "<br>- ".join(
                    analyze_boolean_column(self.df_train, column_name)
                ),
            }

        # type category:
        # - missing values, #unique, most occuring
        if self.type_conversion[column_name]["new"] == "category":
            # Column-level stats
            missing_count = self.df_train[column_name].isna().sum()
            missing_pct = missing_count / len(self.df_train) * 100

            unique_count = self.df_train[column_name].nunique(dropna=True)
            unique_pct = unique_count / len(self.df_train) * 100

            # Frequency table
            frequency = get_frequency_table(self.df_train, column_name)

            # Mode count
            mode_value = self.df_train[column_name].mode(dropna=True)
            mode_count = (
                self.df_train[column_name].value_counts(dropna=True).iloc[0]
                if not frequency == ""
                else 0
            )

            # Entropy
            value_counts = self.df_train[column_name].value_counts(
                normalize=True, dropna=True
            )
            entropy_value = scipy_entropy(value_counts, base=2)  # base-2: bits

            # Cardinality label
            if unique_count <= 5:
                cardinality_label = "Low"
            elif unique_count <= 20:
                cardinality_label = "Medium"
            else:
                cardinality_label = "High"
            suggestions = "- " + "<br>- ".join(
                analyze_categorical_column(self.df_train, column_name)
            )

            return {
                "type": f"category  ({self.type_conversion[column_name]['actual']}) - was {self.type_conversion[column_name]['original']}",
                "description": self.dict_description.get(column_name, ""),
                "missing_values": f"{missing_count} ({missing_pct:.1f}%)",
                "unique_count": f"{unique_count} ({unique_pct:.1f}%)",
                "frequency": frequency,
                "mode": f"{mode_value.iloc[0]} ({mode_count})",
                "entropy": f"{entropy_value:.4f}",
                "cardinality": cardinality_label,
                "suggestions": suggestions,
            }

        # type string:
        # - missing values, #unique, most occuring
        if self.type_conversion[column_name]["new"] == "string":
            missing_count = self.df_train[column_name].isna().sum()
            missing_pct = missing_count / len(self.df_train) * 100
            unique_count = self.df_train[column_name].nunique(dropna=True)
            unique_pct = unique_count / len(self.df_train) * 100
            return {
                "type": f"string  ({self.type_conversion[column_name]['actual']}) - was {self.type_conversion[column_name]['original']}",
                "description": self.dict_description.get(column_name, ""),
                "missing_values": f"{missing_count} ({missing_pct:.1f}%)",
                "unique_count": f"{unique_count} ({unique_pct:.1f}%)",
                "suggestions": "- "
                + "<br>- ".join(
                    analyze_string_column(
                        self.df_train,
                        column_name,
                    )
                ).replace("- #", "&nbsp;&nbsp;"),
            }

        # type numeric:
        # - missing values, #unique, min, max, average, graph
        if self.type_conversion[column_name]["new"] in ["integer", "float"]:
            missing_count = self.df_train[column_name].isna().sum()
            missing_pct = missing_count / len(self.df_train) * 100
            col_data = self.df_train[column_name]
            col_non_null = col_data.dropna()
            return {
                "type": f"{self.type_conversion[column_name]["new"]}  ({self.type_conversion[column_name]['actual']}) - was {self.type_conversion[column_name]['original']}",
                "description": self.dict_description.get(column_name, ""),
                "missing_values": f"{missing_count} ({missing_pct:.1f}%)",
                "min": col_non_null.min() if not col_non_null.empty else None,
                "max": col_non_null.max() if not col_non_null.empty else None,
                "skewness": (
                    skew(col_non_null) if len(col_non_null) > 2 else None
                ),
                "mean": col_non_null.mean() if len(col_non_null) > 2 else None,
                "std_dev": (
                    col_non_null.std() if len(col_non_null) > 2 else None
                ),
                "suggestions": (
                    "- "
                    + "<br>- ".join(
                        analyze_numeric_column(self.df_train, column_name)
                        + analyze_target(self.df_train, column_name)
                    )
                    if column_name == self.target
                    else (
                        "- "
                        + "<br>- ".join(
                            analyze_numeric_column(self.df_train, column_name)
                        )
                    )
                ),
            }

    def analyse_columns(self) -> None:
        assert (
            self.df_train is not None
        ), "Training DataFrame is not initialized"
        self.type_conversion = {}
        total = len(self.df_train.columns)

        for i, column in enumerate(self.df_train.columns, start=1):
            sameline = "[SAMELINE]" if i != 1 or self.nogui else ""
            if i % 2 == 0:
                self.logger.info(
                    f"{sameline}[GREEN]Analyzing columns ({i}/{total})"
                )
            self.type_conversion[column] = {}
            self.type_conversion[column]["original"] = infer_dtype(
                self.df_train[column]
            )
            self.column_type(column)
            self.type_conversion[column]["new"] = infer_dtype(
                self.df_train[column]
            )
        sameline = "[FLUSH]" if self.nogui else "[SAMELINE]"
        self.logger.info(f"{sameline}[GREEN]- Column analysis completed.")
        self.column_info = {}
        self.target_info = {}

        for i, column in enumerate(self.df_train.columns, start=1):
            self.logger.info(
                f"{'[SAMELINE]' if i != 1 or self.nogui else ''}[GREEN]Generating EDA ({i}/{total})"
            )
            self.column_info[column] = {}
            self.column_info[column]["table"] = self.analyse_column(column)
            self.column_info[column]["plots"] = generate_eda_plots(
                self.df_train,
                column,
                infer_dtype(self.df_train[column]),
                self.target,
            )
        sameline = "[FLUSH]" if self.nogui else "[SAMELINE]"
        self.logger.info(f"{sameline}[GREEN]- EDA data completed.")

    def perform_eda(self) -> str:
        project = self.title if self.title else "dataset"
        self.logger.info("[MAGENTA]\nStarting EDA (Exploratory Data Analysis)")
        self.logger.info(f"[MAGENTA]for {project}")
        self.df_train = self.read_data(self.file_train)
        if self.update_script != "":
            self.df_train = self.update_with_user_function(
                self.df_train, self.update_script
            )
        if self.df_train is None:
            self.logger.error("Training data could not be loaded.")
            return "Failed to load training data. EDA cannot proceed."
        last_column = False
        if self.target == "":
            self.target = self.df_train.columns[-1]
            last_column = True
        self.df_test = self.read_data(self.file_test)
        if self.df_test is None:
            self.logger.warning(
                "Test data could not be loaded. Using only training data."
            )
        else:
            if self.update_script != "":
                self.df_test = self.update_with_user_function(
                    self.df_test, self.update_script
                )
            training_columns = set(self.df_train.columns)
            test_columns = set(self.df_test.columns)
            if len(training_columns - test_columns) != 1:
                self.logger.warning(
                    "Test data does not contain all training columns.\n"
                    "EDA will proceed with the available columns."
                )
            else:
                new_target = (training_columns - test_columns).pop()
                if new_target != self.target and not last_column:
                    self.logger.warning(
                        f"Target column '{self.target}' is in test data. "
                        f"Using '{new_target}' as target for EDA."
                    )
                self.target = new_target
        self.dict_description = self.read_description(self.description)
        self.logger.debug(self.dict_description)
        self.analyse_columns()
        target_type = infer_dtype(self.df_train[self.target])
        self.logger.info("[GREEN]- Creating overview table")
        overview_html = create_overview_table(
            df=self.df_train,
            target=self.target,
            target_type=target_type,
            logger=self.logger,
        )
        features_html = get_html_from_template("features.j2", self.column_info)
        self.logger.info("[GREEN]- Getting feature relations")
        self.relation_info, self.num_feats = generate_feature_relations(
            self.df_train,
            self.target,
            self.dict_description,
            logger=self.logger,
        )

        suggestion_overview = {}
        for column in self.df_train.columns:
            summary_suggestions = []
            suggestion_overview[column] = {}
            for cur_suggestion in (
                self.relation_info.get(column, {})
                .get("suggestions", "")
                .split("<br>")
            ):
                if "correlated" in cur_suggestion:
                    summary_suggestions = [cur_suggestion]
            if self.type_conversion[column]["new"] == "string":
                summary_suggestions += self.column_info[column]["table"][
                    "suggestions"
                ].split("<br>")
            if summary_suggestions:
                suggestion_overview[column]["recommendations"] = "<br>".join(
                    summary_suggestions
                )
            else:
                del suggestion_overview[column]

        relation_context = {}
        relation_context["relation_info"] = self.relation_info

        plot1, plot2, shown_feats = generate_relation_visuals(
            self.df_train, self.target
        )
        relation_context["plot1"] = plot1
        relation_context["plot2"] = plot2
        relation_context["num_feats"] = shown_feats

        relations_html = get_html_from_template(
            "relations.j2", relation_context
        )
        self.logger.info("[GREEN]- Getting missing data info.")
        missing_count, column_info_html, general_info_html = (
            missing_data_summary(self.df_train)
        )
        missing_summary = generate_missing_summary(
            self.df_train,
        )
        missing_context = {
            "feature_info": column_info_html,
            "general_info": general_info_html,
            "plot_missing": plot_missingness_matrix(self.df_train, top_n=20),
            "plot_missing_correlation": plot_missing_correlation(self.df_train),
            "missing_summary": missing_summary,
            "missing_data": (missing_count > 0),
        }
        missing_html = get_html_from_template("missing.j2", missing_context)
        # Prepare tab content
        tabs = [
            {"title": "General overview", "content": overview_html},
            {"title": "Features", "content": features_html},
            {"title": "Relations", "content": relations_html},
            {
                "title": "Missing values",
                "content": missing_html,
            },
        ]
        styling = ""
        if self.df_test is not None:
            # Samples
            n_rows = len(self.df_test)
            samples_head = (
                self.df_test.head(10)
                .to_html(index=False, na_rep="<N/A>")
                .replace('border="1"', "")
            )
            samples_middle = (
                self.df_test.iloc[n_rows // 2 - 5 : n_rows // 2 + 5]
                .to_html(index=False, na_rep="<N/A>")
                .replace('border="1"', "")
            )
            samples_tail = (
                self.df_test.tail(10)
                .to_html(index=False, na_rep="<N/A>")
                .replace('border="1"', "")
            )

            # --- Context dictionary for rendering ---

            styling, testdata_content = analyze_test_data(
                self.df_train, self.df_test, self.target
            )
            self.logger.info("[GREEN]- Test data info retrieved.")
            context = {
                "tables": testdata_content,
                "samples_head": samples_head,
                "samples_middle": samples_middle,
                "samples_tail": samples_tail,
            }
            testdata_html = get_html_from_template("testdata.j2", context)
            tabs.append(
                {"title": "Test data", "content": testdata_html},
            )

        recomm_context = {
            "summary": suggestion_overview,
        }
        recomm_html = get_html_from_template(
            "recommendations.j2", recomm_context
        )
        tabs.append(
            {"title": "Recommendations", "content": recomm_html},
        )

        # Load and render the template
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("eda_report.j2")
        output_html = template.render(
            tabs=tabs,
            title=f"EDA Report {self.title}",
            current_time=datetime.now(),
            show_test_data=self.df_test is not None,
            missing_data=(missing_count > 0),
            styling=styling,
        )

        # Save to file
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write(output_html)
        self.logger.info("[BLUE]EDA report saved to %s", self.report_file)
        self.logger.info("[MAGENTA]EDA (Exploratory Data Analysis) is done")
        return "EDA completed successfully with the provided datasets."
