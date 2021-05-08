import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def fit_linear_regression(X, y):
    """
    implements linear regression by solving normal equations.
    :param X: the design matrix - a numpy array with m rows and d columns
    :param y: response vector - numpy array with m rows
    :return: numpy array "w" which represents the coefficients vector with 1 column and and an
    array holding the singular values of the design matrix
    """
    U, s, V_trans = np.linalg.svd(X)  # calc SVD values
    X_dagger = np.linalg.pinv(X)
    return X_dagger @ y, s


def predict(X, w):
    """
    :param X: the design matrix - a numpy array with m rows and d columns
    :param w: coefficients vector
    :return: A numpy array with the predicted values by the model.
    """
    return X @ w


def mse(y, pred_vec):
    """
    calculates the error over given data set
    :param y: response vector
    :param pred_vec: prediction vector
    :return: the MSE over the received samples.
    """
    return np.square(y - pred_vec).mean()


def load_data(csv_file_path):
    """
    load the data set and performs all the needed preprocessing so to get
    a valid design matrix.
    :param csv_file_path:
    :return: the dataset after preprocessing.
    """
    df = pd.read_csv(csv_file_path)
    pearson_corr = np.abs(df.corr())  # get correlation matrix between all columns
    df.dropna(inplace=True)  # remove all rows with nan

    # drop useless columns
    drop_features = ["long", "lat", "id", "date"]
    df.drop(drop_features, inplace=True, axis=1)

    # drop rows with wrong/illogical data
    pos_features = ["price", "bedrooms", "bathrooms", "sqft_living", "yr_built"]
    for feature in pos_features:
        to_drop = np.argwhere(np.array(df[feature]) <= 0).T[0].tolist()
        df.drop(to_drop, inplace=True)

    categorical = ["zipcode"]
    df = pd.get_dummies(df, columns=categorical)
    response_vec = df["price"]  # get response vector
    corr_processing(df, pearson_corr)
    df.pop("price")
    return df, response_vec


def corr_processing(df, pearson_corr):
    """

    :param df:
    :param pearson_corr:
    :return:
    """
    lower, upper = 0.75, 0.9
    high_corr_indices = np.argwhere((lower < np.array(pearson_corr)) & \
                                    (np.array(pearson_corr) < upper))
    index_set = {tuple(item) for item in high_corr_indices if item[0] < item[1]}
    arr = set()
    for pair in index_set:
        if pearson_corr.iloc[1, pair[0]] > pearson_corr.iloc[1, pair[1]]:
            arr.add(df.columns[pair[1]])
        else:
            arr.add(df.columns[pair[0]])
    for col in arr:
        df.pop(col)


def plot_singular_values(singular_arr):
    """
    receives an array of singular values and plots them in descending order
    :param singular_arr: singular values array
    """
    x_axis = np.arange(1, singular_arr.size + 1)
    plt.scatter(x_axis, sorted(singular_arr, reverse=True))
    plt.ylabel("Singular values")
    plt.title("Singular values vs. number of value")
    plt.yscale("log")
    plt.savefig("Singular_val_graph.jpeg")
    plt.show()


def q_16(df, response_vec):
    """
    splits the data into test and train sets where test set is 1/4 of the total
    data. plots MSE as a function of p.
    :param response_vec: the given value to compare against.
    :param df: data frame containing the data.
    :return:
    """
    mse_arr, p_arr = [], []
    train_set, test_set, y_train, y_test = train_test_split(df, response_vec)
    for p in range(1, 101):
        max_row_idx = int(np.floor((p / 100) * train_set.shape[0]))
        temp_X = train_set[:max_row_idx]  # subset to train over
        w_hat = fit_linear_regression(temp_X, y_train[:max_row_idx])[0]
        y_hat = predict(test_set[:max_row_idx], w_hat)
        mse_arr.append(mse(y_test[:max_row_idx], y_hat))
        p_arr.append(p)
    plt.plot(p_arr, mse_arr, color="navy")
    plt.xlabel("% of samples")
    plt.ylabel("MSE")
    plt.title("MSE vs. % of samples")
    plt.savefig("MSE_vs_p2.jpeg")
    plt.show()


def feature_evaluation(X, y):
    """

    :param X:
    :param y:
    :return:
    """
    categorical = ["id", "zipcode", "lat", "long", "date", "waterfront"]
    for category in X:
        if category not in categorical and "zipcode" not in category:
            plt.scatter(X[category], y, c="firebrick")
            plt.title("Pearson-corr: " +
                      "$\\rho=$" + str(np.around(pearson_corr(X[category], y)[0, 1], 3)))
            plt.xlabel(category)
            plt.ylabel("Response")
            plt.savefig("Response vs." + category + ".jpeg")
            plt.show()


def pearson_corr(vec1, vec2):
    """
    calculates the pearson correlation between two given vectors.
    """
    return np.cov(vec1, vec2) / (np.std(vec1) * np.std(vec2))
