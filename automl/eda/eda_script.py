# Local application imports
from automl.dataloader import ConfigData, OriginalData
from automl.library import todo  # only during develloping

from .column_analysis import analyse_columns, insert_descriptions
from .dataset_overview import find_duplicate_columns, analyse_dataset


def perform_eda(config_data: ConfigData, original_data: OriginalData) -> None:

    dict_duplicates = find_duplicate_columns(X_train=original_data.X_train)
    column_info = analyse_columns(
        X_train=original_data.X_train,
        dict_duplicates=dict_duplicates,
        y_train=original_data.y_train,
    )
    column_info = insert_descriptions(
        column_info=column_info, config_data=config_data
    )
    data_set_info = analyse_dataset(
        X_train=original_data.X_train,
        column_info=column_info,
        dict_duplicates=dict_duplicates,
        y_train=original_data.y_train,
    )
    # analyse_relations()
    # analyse_test_data()
    # preprocess_trial()
    # create_report()
    todo()
    return
    return data_set_info
