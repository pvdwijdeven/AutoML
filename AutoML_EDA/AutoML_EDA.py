import pandas as pd


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

    def perform_eda(self):
        self.df_train = self.read_data(self.file_train)
        if self.df_train is None:
            self.logger.error("Training data could not be loaded.")
            return "Failed to load training data. EDA cannot proceed."
        self.df_test = self.read_data(self.file_test)
        if self.df_test is None:
            self.logger.error(
                "Test data could not be loaded. Using only training data."
            )

        print(self.df_train.info())

        if self.df_test is not None:
            print(self.df_test.info())
        return "EDA completed successfully with the provided datasets."
