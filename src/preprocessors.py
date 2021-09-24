import pandas as pd


class Preprocessor:
    """
    Base Preprocessor class
    """
    def __call__(self, data, with_train=True):
        NotImplemented


class Compose(Preprocessor):
    """
    Apply set of preprocessors
    """
    def __init__(self, preprocessors=[]):
        self.preprocessors = preprocessors

    def __call__(self, data, with_train=True):
        for preprocessor in self.preprocessors:
            preprocessor(data, with_train)


class LabelEncoder(Preprocessor):
    """
    Encodes cat features to
    """
    def __init__(self,
                 label_names,
                 label_encoder_class,
                 label_encoder_kwargs):
        super().__init__()
        self.encoders = {}
        self.label_names = label_names
        for label in self.label_names:
            self.encoders[label] = label_encoder_class(**label_encoder_kwargs)

    def __call__(self, data, with_train=True):
        if with_train:
            for label in self.label_names:
                self.encoders[label].fit(data[label])
        for label in self.label_names:
            data[label] = self.encoders[label].transform(data[label])


class ColumnDropper(Preprocessor):
    """
    Drop columns
    """
    def __init__(self, column_names):
        self.column_names = column_names

    def __call__(self, data, with_train=True):
        data.drop(columns=self.column_names, inplace=True)


class TimeFeaturesExtractor(Preprocessor):
    """
    Extract features from timestamp
    """
    def __init__(self, column_names):
        self.column_names = column_names

    def __call__(self, data, with_train=True):
        for column_name in self.column_names:
            datetime_ = pd.to_datetime(data[column_name]).dt
            data[f"{column_name} (YEAR)"] = datetime_.year
            data[f"{column_name} (MONTH)"] = datetime_.month
            data[f"{column_name} (DAY)"] = datetime_.day
            data[f"{column_name} (HOUR)"] = datetime_.hour
            data[f"{column_name} (MINUTE)"] = datetime_.minute


class DurationExtractor(Preprocessor):
    """
    Extract duration of flight
    """
    def __init__(self, time_from_column_name, time_to_column_name):
        self.time_from_column_name = time_from_column_name
        self.time_to_column_name = time_to_column_name

    def __call__(self, data, with_train=True):
        datetime_from = pd.to_datetime(data[self.time_from_column_name])
        datetime_to = pd.to_datetime(data[self.time_to_column_name])

        data["Duration"] = (datetime_to - datetime_from).astype('timedelta64[m]')


class TrainTestIdentifier(Preprocessor):
    """
    Split on train test
    """
    def __init__(self, time_from_column_name):
        self.time_from_column_name = time_from_column_name

    def __call__(self, data, with_train=True):
        datetime_ = pd.to_datetime(data[self.time_from_column_name]).dt
        data["is_train"] = datetime_.year < 2018