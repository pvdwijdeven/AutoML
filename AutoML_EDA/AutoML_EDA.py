import pandas as pd


class AutoML_EDA:
    def __init__(
        self,
        file_train,
        file_test="",
    ) -> None:
        self.file_train = file_train
        self.file_test = file_test

        self.df_train = self.read_data(file_train)
        if self.df_train is None:
            raise ValueError("Training data could not be loaded.")

        self.df_test = self.read_data(file_test)
        if self.df_test is None:
            raise ValueError("Test data could not be loaded.")

    def read_data(self, file_path) -> pd.DataFrame | None:
        if not file_path:
            return None
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def perform_eda(self):
        assert self.df_train is not None, "Training data is not loaded."
        assert self.df_test is not None, "Test data is not loaded."
        print(self.df_train.info())
        print(self.df_test.info())
