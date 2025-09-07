# Local application imports
from automl.dataloader import ConfigData, OriginalData
from automl.library import todo  # only during develloping

from .column_analysis import analyse_columns, insert_descriptions
from .dataset_overview import analyse_dataset, find_duplicate_columns
from .eda_report import create_report
from .relations import generate_feature_relations
from .testdata import analyze_test_data


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

    _relation_info = generate_feature_relations(original_data=original_data)
    _test_info = analyze_test_data(original_data=original_data)

    # preprocess_trial() # todo once preprocessing is ready

    create_report(
        config_data,
        data_set_info,
        original_data=original_data,
        column_info=column_info,
    )
    todo()
    return
