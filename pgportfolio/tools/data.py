from __future__ import division,absolute_import,print_function
import numpy as np
import pandas as pd
from traceback import print_stack
import logging
import time
import math

def pricenorm3d(m, features, norm_method, fake_ratio=1.0, with_y=True):
    """normalize the price tensor, whose shape is [features, coins, windowsize]
    @:param m: input tensor, unnormalized and there could be nan in it
    @:param with_y: if the tensor include y (future price)
        logging.debug("price are %s" % (self._latest_price_matrix[0, :, -1]))
    """
    result = m.copy()
    if features[0] != "close":
        raise ValueError("first feature must be close")
    for i, feature in enumerate(features):
        if with_y:
            one_position = 2
        else:
            one_position = 1
        pricenorm2d(result[i], m[0, :, -one_position], norm_method=norm_method,
                    fake_ratio=fake_ratio, one_position=one_position)
    return result


# input m is a 2d matrix, (coinnumber+1) * windowsize
def pricenorm2d(m, reference_column,
                norm_method="absolute", fake_ratio=1.0, one_position=2):
    if norm_method=="absolute":
        output = np.zeros(m.shape)
        for row_number, row in enumerate(m):
            if np.isnan(row[-one_position]) or np.isnan(reference_column[row_number]):
                row[-one_position] = 1.0
                for index in range(row.shape[0] - one_position + 1):
                    if index > 0:
                        row[-one_position - index] = row[-index - one_position + 1] / fake_ratio
                row[-one_position] = 1.0
                row[-1] = fake_ratio
            else:
                row = row / reference_column[row_number]
                for index in range(row.shape[0] - one_position + 1):
                    if index > 0 and np.isnan(row[-one_position - index]):
                        row[-one_position - index] = row[-index - one_position + 1] / fake_ratio
                if np.isnan(row[-1]):
                    row[-1] = fake_ratio
            output[row_number] = row
        m[:] = output[:]
    elif norm_method=="relative":
        output = m[:, 1:]
        divisor = m[:, :-1]
        output = output / divisor
        pad = np.empty((m.shape[0], 1,))
        pad.fill(np.nan)
        m[:] = np.concatenate((pad, output), axis=1)
        m[np.isnan(m)] = fake_ratio
    else:
        raise ValueError("there is no norm morthod called %s" % norm_method)


def get_chart_until_success(exchange, pair, start, period, end):
    is_connect_success = False
    chart = {}
    logging.error('get_chart_until_success: exchange='+str(exchange)+' pair='+pair+' start='+str(start)+' period='+str(period)+' end='+str(end))
    while not is_connect_success:
        try:
            chart = exchange.marketChart(pair=pair, start=int(start), period=int(period), end=int(end))
            is_connect_success = True
        except Exception as e:
            print(e)
            logging.error('Exchange connection failed: ' + str(e))
            print_stack()
#    logging.error('get_chart_until_success returning: ' + str(chart))
    return chart


def get_type_list(feature_number):
    """
    :param feature_number: an int indicates the number of features
    :return: a list of features n
    """
    if feature_number == 1:
        type_list = ["close"]
    elif feature_number == 2:
        type_list = ["close", "volume"]
#        print_stack()
        raise NotImplementedError("the feature volume is not supported currently")
    elif feature_number == 3:
        type_list = ["close", "high", "low"]
    elif feature_number == 4:
        type_list = ["close", "high", "low", "open"]
    else:
        raise ValueError("feature number could not be %s" % feature_number)
    return type_list


def panel2array(panel):
    """convert the panel to datatensor (numpy array) without btc
    """
    without_btc = np.transpose(panel.values, axes=(2, 0, 1))
    return without_btc


def count_periods(start, end, period_length):
    """
    :param start: unix time, excluded
    :param end: unix time, included
    :param period_length: length of the period
    :return: 
    """
    return (int(end)-int(start)) // period_length


def get_volume_forward(time_span, portion, portion_reversed):
    volume_forward = 0
    if not portion_reversed:
        volume_forward = time_span*portion
    return volume_forward


def panel_fillna(panel, type="bfill"):
    """
    fill nan along the 3rd axis
    :param panel: the panel to be filled
    :param type: bfill or ffill
    """
    start_ts = time.time()
#    print ("panel_fillna: called.")
    frames = {}
#    return panel
    for item in panel.items:
#        frames[item] = panel.loc[item].fillna(panel.loc[item].mean(axis=0, level=-1)) # With 0 - div/0 error. With mean() - hangs.
#        frames[item] = panel.loc[item].fillna(1.) # No hang, but loss=-2e-1 lol. "the portfolio value on test set is inf"
#        continue
        if type == "both":
            frames[item] = panel.loc[item].fillna(axis=1, method="bfill").\
                fillna(axis=1, method="ffill")
        else:
            frames[item] = panel.loc[item].fillna(axis=1, method=type)
#    print("panel_fillna: took " + str(int(time.time() - start_ts)) + " seconds")
    return pd.Panel(frames)

def panel_remove_outliers(panel, threshold=1.):
    start_ts = time.time()
    logging.error ("\n\nPanel shape: " + str(panel) + "\n\n")
    fh = {} # future history
    return panel
    for date in reversed (panel.minor_axis) :
#        print ("Date (key): " + str(date))
#        items = panel.minor_xs (date) # .items();
        coins = panel.minor_xs (date).transpose()
        for coin in coins:
#            if panel.at['close', coin, date] == 0.:
#                continue
            if math.isnan (panel.at['close', coin, date]) or panel.at['high', coin, date] / panel.at['low', coin, date] > 1. + threshold:
                if not math.isnan (panel.at['close', coin, date]):
                    logging.error ('Outlier -- Coin: ' + str(coin) + ' Date: ' + str(date) + ' High: ' + str(panel.at['high', coin, date]) + ' Low: ' + str(panel.at['low', coin, date]) + ' Close: ' + str(panel.at['close', coin, date]))
                    logging.error ('        -- Changed to [low: ' + str(fh[coin]['low']) + ', high: ' + str(fh[coin]['high']) + ', close: ' + str(fh[coin]['close']) + ']')
                panel.at['low', coin, date] = fh[coin]['low']
                panel.at['high', coin, date] = fh[coin]['high']
                panel.at['close', coin, date] = fh[coin]['close']

            fh[coin] = {'low': panel.at['low', coin, date], 'high': panel.at['high', coin, date], 'close': panel.at['close', coin, date]}
#            if 
#            print ("Coin: " + str(coin));
#            print ("Data: " + str(panel.at['close', coin, date]))
#            print ("Data: " + str(panel.minor_xs (date).items (coin)))
#        print (panel.minor_xs (key))
    logging.error ("panel_remove_outliers: took " + str(int(time.time() - start_ts)) + " seconds")
    return panel
