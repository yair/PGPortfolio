from __future__ import division,absolute_import,print_function
import numpy as np
from pgportfolio.marketdata.datamatrices import DataMatrices
from pgportfolio.marketdata.globaldatamatrix import HistoryManager
from pgportfolio.tools.configprocess import parse_time
from pgportfolio.constants import *
from pgportfolio.tools.data import get_volume_forward
from time import time


def get_coin_name_list(config, online):
    """
    :param online: boolean value to show if connected to internet,
    if False, load data from database.
    :return : list of coin names
    """
    assert False # Never called
    input_config = config["input"]
    if not online:
        start = parse_time(input_config["start_date"])
        end = parse_time(input_config["end_date"])
        volume_forward = get_volume_forward(end - start,
                                            input_config["test_portion"]
                                            + input_config["validation_portion"],
                                            input_config["portion_reversed"])
    else:
        end = time()
        volume_forward = 0
    end = end - (end % input_config["trade_period"])
    start = end - volume_forward - input_config["volume_average_days"] * DAY
    end = end - volume_forward
    coins = HistoryManager(input_config["market"], input_config["coin_number"], end,
                           volume_forward=volume_forward,
                           volume_average_days=input_config["volume_average_days"],
                           online=online).\
        select_coins(start, end)
    return coins


def calculate_pv_after_commission_cv(w1, w0, cv):
    """
    @:param w1: target portfolio vector, first element is btc
    @:param w0: rebalanced last period portfolio vector, first element is btc
    @:param commission_rate: rate of commission fee, proportional to the transaction cost

    'k. Dimensinal analysis.
    Originally, w0, w1 are c+1 vectors and commission_rate is a scalar, for us w0, w1 are c+1 vectors, and cv is a c-vector.
    The mus need to be scalar (the return is defly scalar). So what's the equivalent of (1-2x+x^2)?
    The quadratic term is negligible, but it's not a problem as x*x is a scalar via a dot product anyway.
    In nnagent, we use mu = 1 - tf.reduce_sum(tf.matmul(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv), axis=1) or 1-sum(|w1-w0| X cv)
    Is the quadratic part the only reason for this approximation loop?
    mu0=1
    mu1=1-2c+c^2
    while (|mu1-mu0| > 1e-10)
        mu0=mu1
        mu1=(1-c*w0[0]-(2c-c^2)sum(max(w0[1:]-mu1*w1[1:], 0)))/(1-cw1[0])
           =(1-cw0[0])/(1-cw1[0])
            change in BTCp
    I don't understand.
    Let's just try copy paste.
    """
    return 1 - np.sum(np.matmul(abs(w1[1:]-w0[1:]), cv))
    """
    mu0 = 1
    mu1 = 1 - 2*commission_rate + commission_rate ** 2
    while abs(mu1-mu0) > 1e-10:
        mu0 = mu1
        mu1 = (1 - commission_rate * w0[0] -
            (2 * commission_rate - commission_rate ** 2) *
            np.sum(np.maximum(w0[1:] - mu1*w1[1:], 0))) / \
            (1 - commission_rate * w1[0])
    return mu1
    """

def calculate_pv_after_commission(w1, w0, commission_rate):
    """
    @:param w1: target portfolio vector, first element is btc
    @:param w0: rebalanced last period portfolio vector, first element is btc
    @:param commission_rate: rate of commission fee, proportional to the transaction cost
    """
    assert False
    """
    mu0 = 1
    mu1 = 1 - 2*commission_rate + commission_rate ** 2
    while abs(mu1-mu0) > 1e-10:
        mu0 = mu1
        mu1 = (1 - commission_rate * w0[0] -
            (2 * commission_rate - commission_rate ** 2) *
            np.sum(np.maximum(w0[1:] - mu1*w1[1:], 0))) / \
            (1 - commission_rate * w1[0])
    return mu1
    """

def get_test_data(config):
    """
    :return : a 2d numpy array with shape(coin_number, periods),
     each element the relative price
    """
    config["input"]["feature_number"] = 1
    config["input"]["norm_method"] = "relative"
    config["input"]["global_period"] = config["input"]["global_period"]
    price_matrix = DataMatrices.create_from_config(config)
    test_set = price_matrix.get_test_set()["y"][:, 0, :].T
    test_set = np.concatenate((np.ones((1, test_set.shape[1])), test_set), axis=0)
    return test_set

def asset_vector_to_dict(coin_list, vector, with_BTC=True):
    vector = np.squeeze(vector)
    dict_coin = {}
    if with_BTC:
        dict_coin['BTC'] = vector[0]
    for i, name in enumerate(coin_list):
        if vector[i+1] > 0:
            dict_coin[name] = vector[i + 1]
    return dict_coin

def save_test_data(config, file_name="test_data", output_format="csv"):
    if output_format == "csv":
        matrix = get_test_data(config)
        with open(file_name+"."+output_format, 'wb') as f:
            np.savetxt(f, matrix.T, delimiter=",")

