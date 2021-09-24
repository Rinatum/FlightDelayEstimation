import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse,\
                            mean_absolute_error as mae


from matplotlib import pyplot as plt
from scipy.stats import zscore


def plot_hist(feature_value_counts,
              figsize={'figure.figsize': (16, 50)},
              title="some title"):
    sns.set(rc=figsize)
    sns.barplot(y=feature_value_counts.index,
                x=feature_value_counts,
                orient='h').set_title(title)
    plt.show(sns)


def split_train_test(data):
    train = data[data["is_train"]].drop(columns=["is_train"])
    valid = data[~data["is_train"]].drop(columns=["is_train"])

    return train, valid


def plot_dependencies(data, target_name,
                      title="Feature - Target dependencies",
                      figsize=(32,16),
                      fontsize=32):
    features = list(data.columns)
    features.remove(target_name)
    rows = int(np.sqrt(len(features))) + 1
    fig, axes = plt.subplots(rows, rows, figsize=figsize)
    fig.suptitle(title, fontsize=fontsize)
    for feature_name, ax in zip(features, axes.reshape(-1)):
        sns.scatterplot(ax=ax, data=data, x=feature_name, y=target_name)


def compute_metrics(train_pred, train_target, valid_pred, valid_target):
    metrics = {"MAE": mae,
               "MSE": mse,
               "RMSE": lambda pred, target: mse(target, pred, squared=False)}

    for metric_name, metric in metrics.items():
        print(f"Train {metric_name} = {metric(train_target, train_pred)}")

    for metric_name, metric in metrics.items():
        print(f"Valid {metric_name} = {metric(valid_target, valid_pred)}")


def remove_outliers_by_z_score(train,
                               train_target,
                               tresh=3):
    confident_train = (np.abs(zscore(train)) < tresh).all(axis=1)
    cleaned_train = train[confident_train]
    cleaned_train_target = train_target[confident_train]

    return cleaned_train, cleaned_train_target



