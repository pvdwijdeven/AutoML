import pandas as pd
import json
import pprint
from jinja2 import Environment, FileSystemLoader
from .AutoML_EDA_overview import create_overview_table
from AutoML_Libs import infer_dtype


class AutoML_EDA:
    def __init__(
        self,
        logger,
        file_train,
        file_test="",
    ) -> None:
        self.file_train = file_train
        self.file_test = file_test
        self.logger = logger

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
            print(f"Error reading {file_path}: {e}")
            return None

    def column_type(
        self,
        column_name: str,
    ) -> None:
        def normalize_booleans(series: pd.Series) -> pd.Series:
            if pd.api.types.is_object_dtype(
                series
            ) or pd.api.types.is_string_dtype(series):
                return series.map(
                    lambda x: str(x).strip().lower() if pd.notna(x) else x
                ).replace({"true": True, "false": False})
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
        col_series_train = normalize_booleans(col_series_train)

        # Step 2: Detect if boolean
        unique_values = pd.Series(col_series_train.dropna().unique())
        if unique_values.isin([True, False]).all():
            self.df_train[column_name] = col_series_train.astype("boolean")
            if self.df_test is not None and column_name in self.df_test.columns:
                self.df_test[column_name] = normalize_booleans(
                    self.df_test[column_name]
                ).astype("boolean")
            return

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
        elif pd.api.types.is_numeric_dtype(col_series_train):
            converted = col_series_train.copy()
        else:
            converted = None

        if converted is not None:
            non_na_count = col_series_train.notna().sum()
            if converted.notna().sum() == non_na_count:
                if (
                    pd.api.types.is_float_dtype(converted)
                    and (converted.dropna() % 1 == 0).all()
                ):
                    self.df_train[column_name] = converted.astype("Int64")
                    inferred_dtype = "Int64"
                else:
                    self.df_train[column_name] = converted
                    inferred_dtype = str(converted.dtype)

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
                "float"
            ):
                try:
                    converted_test = convert_to_numeric(col_test)
                    # Use numpy dtype object for astype
                    import numpy as np

                    self.df_test[column_name] = converted_test.astype(
                        np.float64
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to apply float conversion to test column '{column_name}': {e}"
                    )

    def analyse_column(self, column_name: str):

        assert (
            self.df_train is not None
        ), "Training DataFrame is not initialized"

        # type boolean:
        # - missing values, frequency True/False, graph
        if self.type_conversion[column_name]["new"] == "boolean":
            missing_count = self.df_train[column_name].isna().sum()
            missing_pct = missing_count / len(self.df_train) * 100
            return {
                "type": "boolean",
                "missing_values": missing_count,
                "missing_percentage": missing_pct,
                "frequency": self.df_train[column_name]
                .value_counts()
                .to_dict(),
            }

        # type category:
        # - missing values, #unique, most occuring
        if self.type_conversion[column_name]["new"] == "category":
            missing_count = self.df_train[column_name].isna().sum()
            missing_pct = missing_count / len(self.df_train) * 100
            unique_count = self.df_train[column_name].nunique(dropna=True)
            unique_pct = unique_count / len(self.df_train) * 100
            return {
                "type": "category",
                "missing_values": missing_count,
                "missing_percentage": missing_pct,
                "unique_count": unique_count,
                "unique_percentage": unique_pct,
                "frequency": self.df_train[column_name]
                .value_counts()
                .to_dict(),
            }

        # type string:
        # - missing values, #unique, most occuring
        if self.type_conversion[column_name]["new"] == "string":
            missing_count = self.df_train[column_name].isna().sum()
            missing_pct = missing_count / len(self.df_train) * 100
            unique_count = self.df_train[column_name].nunique(dropna=True)
            unique_pct = unique_count / len(self.df_train) * 100
            return {
                "type": "category",
                "missing_values": missing_count,
                "missing_percentage": missing_pct,
                "unique_count": unique_count,
                "unique_percentage": unique_pct,
            }

        # type numeric:
        # - missing values, #unique, min, max, average, graph
        if self.type_conversion[column_name]["new"] in ["integer", "float"]:
            missing_count = self.df_train[column_name].isna().sum()
            missing_pct = missing_count / len(self.df_train) * 100
            return {
                "type": self.type_conversion[column_name]["new"],
                "missing_values": missing_count,
                "missing_percentage": missing_pct,
                "min": self.df_train[column_name].min(),
                "max": self.df_train[column_name].max(),
                "mean": self.df_train[column_name].mean(),
                "std_dev": self.df_train[column_name].std(),
            }

    def analyse_columns(self) -> None:
        assert (
            self.df_train is not None
        ), "Training DataFrame is not initialized"
        self.type_conversion = {}
        for column in self.df_train.columns:
            self.type_conversion[column] = {}
            self.type_conversion[column]["original"] = infer_dtype(
                self.df_train[column]
            )
            self.column_type(column)
            self.type_conversion[column]["new"] = infer_dtype(
                self.df_train[column]
            )
        self.logger.info(
            "[GREEN]- Column type analysis completed. Changes made:"
        )
        self.logger.info(f"[CYAN]{json.dumps(self.type_conversion, indent=4)}")
        self.column_info = {}
        for column in self.df_train.columns:
            self.column_info[column] = self.analyse_column(column)
        self.logger.info("[GREEN]- Column analysis completed. Details:")
        self.logger.info(f"[CYAN]{pprint.pformat(self.column_info)}")

    def perform_eda(self) -> str:
        self.logger.info("[MAGENTA]Starting EDA (Exploratory Data Analysis)")
        self.df_train = self.read_data(self.file_train)
        if self.df_train is None:
            self.logger.error("Training data could not be loaded.")
            return "Failed to load training data. EDA cannot proceed."
        self.df_test = self.read_data(self.file_test)
        if self.df_test is None:
            self.logger.error(
                "Test data could not be loaded. Using only training data."
            )

        print(f"training data: {self.df_train.shape}")

        if self.df_test is not None:
            print(f"test data: {self.df_test.shape}")

        self.analyse_columns()
        overview_html = create_overview_table(df=self.df_train)

        # Prepare tab content
        tabs = [
            {"title": "General overview", "content": overview_html},
            {"title": "Features", "content": ""},
            {"title": "Target", "content": ""},
            {"title": "Relations", "content": ""},
            {"title": "Missing values", "content": ""},
        ]

        # Load and render the template
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("EDA_report.html")
        output_html = template.render(tabs=tabs, title="EDA Report")

        # Save to file
        with open("export\\output.html", "w", encoding="utf-8") as f:
            f.write(output_html)
        self.logger.info("[MAGENTA]EDA (Exploratory Data Analysis) is done")
        return "EDA completed successfully with the provided datasets."
