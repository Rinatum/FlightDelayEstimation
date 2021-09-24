import argparse

import pandas as pd
from sklearn import preprocessing
from preprocessors import \
    TimeFeaturesExtractor, \
    DurationExtractor, \
    TrainTestIdentifier, \
    LabelEncoder, \
    ColumnDropper, \
    Compose
from utils import split_train_test, remove_outliers_by_z_score
from experiments import experiment_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", help="Path to csv file")

    args = parser.parse_args()
    data = pd.read_csv(args.csv_path)

    preprocessor = Compose(
        [
            TimeFeaturesExtractor(["Scheduled depature time", "Scheduled arrival time"]),
            DurationExtractor("Scheduled depature time", "Scheduled arrival time"),
            LabelEncoder(["Depature Airport",
                          "Destination Airport",
                          "Scheduled depature time (YEAR)",
                          "Scheduled arrival time (YEAR)"
                          ], preprocessing.LabelEncoder, {}),
            TrainTestIdentifier("Scheduled depature time"),
            ColumnDropper(["Scheduled depature time", "Scheduled arrival time"])
        ])

    preprocessed_data = data.copy()
    preprocessor(preprocessed_data, with_train=True)

    train, test = split_train_test(preprocessed_data)
    train_target = train["Delay"]
    train.drop(columns=["Delay"], inplace=True)
    test_target = test["Delay"]
    test.drop(columns=["Delay"], inplace=True)

    cleaned_train, cleaned_train_target = \
        remove_outliers_by_z_score(train, train_target, tresh=3)

    experiment_models(cleaned_train,
                      test,
                      cleaned_train_target,
                      test_target)


