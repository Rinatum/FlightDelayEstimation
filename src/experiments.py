from sklearn import preprocessing
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from utils import compute_metrics


def experiment_models(train, test, train_target, test_target):
    """
    Experiments with different models
    :param train:
    :param test:
    :param train_target:
    :param test_target:
    :return:
    """
    # Linear models
    linear_models = [(LinearRegression, {"n_jobs": -1}),
                     (Lasso, {"alpha": 3}),
                     (Ridge, {"alpha": 3}),
                     (LinearSVR, {"random_state": 0, "tol": 1e-5})]

    # Add polynomial features
    poly = preprocessing.PolynomialFeatures(2)

    # scaler
    scaler = preprocessing.StandardScaler().fit(train)

    print("Use linear models with linear features")
    for model_ in linear_models:
        scaled_train = scaler.transform(train)
        scaled_test = scaler.transform(test)
        model = model_[0](**model_[1])
        model.fit(scaled_train, train_target.to_numpy())
        train_pred = model.predict(scaled_train)
        valid_pred = model.predict(scaled_test)
        print("=========================================")
        print(f"Model : {model_}")
        compute_metrics(train_pred, train_target, valid_pred, test_target)
        print("=========================================")

    print("Use linear models with polynomial features")
    train = poly.fit_transform(train)
    test = poly.transform(test)
    scaler = preprocessing.StandardScaler().fit(train)
    for model_ in linear_models:
        scaled_train = scaler.transform(train)
        scaled_test = scaler.transform(test)
        model = model_[0](**model_[1])
        model.fit(scaled_train, train_target.to_numpy())
        train_pred = model.predict(scaled_train)
        valid_pred = model.predict(scaled_test)
        print("=========================================")
        print(f"Model : {model_}")
        compute_metrics(train_pred, train_target, valid_pred, test_target)
        print("=========================================")
