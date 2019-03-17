from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from pgportfolio.marketdata.coinlist import CoinList
import numpy as np
import pandas as pd
from pgportfolio.tools.data import panel_fillna
import pgportfolio.constants as const
import sqlite3
from datetime import datetime
import logging
import re
from traceback import print_stack
import os
import json
import time

class HistoryManager:
    # if offline ,the coin_list could be None
    # NOTE: return of the sqlite results is a list of tuples, each tuple is a row
    def __init__(self, market, coin_number, end, volume_average_days=1, volume_forward=0, online=True, live=False, net_dir="", augment_train_set=False):
        self.market = market
        self.initialize_db()
        self.__storage_period = const.FIVE_MINUTE  # keep this as 300
        self._coin_number = coin_number
        self._online = online
        self._live = live
        self._augment_train_set = augment_train_set
        if net_dir != None and net_dir != '':
            self._net_dir = net_dir.replace("/netfile", "")
            logging.error("HistoryManager: net_dir is at '" + net_dir + "'")
        else:
            self._net_dir = None
            logging.error("HistoryManager: net_dir is nowhere to be found. Is this download_data?")
        if self._online:
            self._coin_list = CoinList(market, end, volume_average_days, volume_forward, live, net_dir)
        self.__volume_forward = volume_forward
        self.__volume_average_days = volume_average_days
        self.__coins = None

    @property
    def coins(self):
        return self.__coins

    def initialize_db(self):
        with sqlite3.connect(const.DATABASE_DIR + '.' + self.market) as connection:
            cursor = connection.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS History (date INTEGER,'
                           ' coin varchar(20), high FLOAT, low FLOAT,'
                           ' open FLOAT, close FLOAT, volume FLOAT, '
                           ' quoteVolume FLOAT, weightedAverage FLOAT,'
                           'PRIMARY KEY (date, coin));')
#                           'PRIMARY KEY (coin, date));')
            connection.commit()

    def get_current_balances(self):
        # Return as an array matching the current coin list
        all_balances = self._coin_list.getBalances()
        balances = [float(all_balances['BTC'])]
        for coin in self.__coins:
#            logging.error("Now getting balance of coin '{}'".format(coin))
            m = re.match('reversed_(\w+)$', coin)
            if (m != None):
#                logging.error("Coin {} is reversed".format(coin))
                coin = m.group(1)
#                logging.error("Now called {}.".format(coin))
            assert coin in all_balances
            balance = float(all_balances[coin])
#            logging.error("Balance {} is of type ".format(balance) + type(balance).__name__)
#            if (m != None):
#                balance = 1. / balance
            balances.append(balance)
        logging.error("get_current_balances: " + str(balances))
        return balances

    def get_global_data_matrix(self, start, end, period=300, features=('close',)):
        """
        :return a numpy ndarray whose axis is [feature, coin, time]
        """
        logging.error("Calling get_global_panel from HistoryManager::get_global_data_matrix")
        return self.get_global_panel(start, end, period, features).values

    def get_aug_factor (self, period=300):
        return period // self.__storage_period

    def get_global_panel(self, start, end, period=300, features=('close',)):
        """
        :param start/end: linux timestamp in seconds
        :param period: time interval of each data access point
        :param features: tuple or list of the feature names
        :return a panel, [feature, coin, time]
        """
        start_ts = time.time()
        logging.error("start: {}".format(start));
        logging.error("period: {}".format(period));
        start = int(start - (start % period))
        end = int(end - (end % period))
        logging.error("get_global_panel called with self from " + self.__class__.__name__ + " (Live session: " + str(self._live) + ")");
#        print_stack()
        coins = self.select_coins(start=end - self.__volume_forward - self.__volume_average_days * const.DAY,
                                  end=end - self.__volume_forward)

        if len(coins) != self._coin_number:
            raise ValueError("the length of selected coins %d is not equal to expected %d"
                             % (len(coins), self._coin_number))

        self.__coins = coins
        for coin in coins:
            self.update_data(start, end, coin)

#        if len(coins) != self._coin_number:
#            raise ValueError("the length of selected coins %d is not equal to expected %d"
#                             % (len(coins), self._coin_number))

        logging.info("get_global_panel: feature type list is %s" % str(features))
        self.__checkperiod(period)

        if self._augment_train_set:
            time_index = pd.to_datetime(list(range(start, end + 1, self.__storage_period)), unit='s')
        else:
            time_index = pd.to_datetime(list(range(start, end + 1, period)), unit='s')
        panel = pd.Panel(items=features, major_axis=coins, minor_axis=time_index, dtype=np.float32)

        logging.error("get_global_panel: Getting data from " + str(start) + " to " + str(end) + " from DB at " + const.DATABASE_DIR + "." + self.market)
        connection = sqlite3.connect(const.DATABASE_DIR + '.' + self.market)
        connection.execute("PRAGMA cache_size = 1000000") # might help. dunno. Also, why do we reconnect each time?
        connection.commit()
        logging.error("get_global_panel: time till big loop: " + str(int(time.time() - start_ts)) + " seconds")
        start_ts = time.time()
        try:
            if self._augment_train_set:
                panel = self.__get_data_augmented (panel, connection, coins, features, start, end, period)
            else:
                panel = self.__get_data (panel, connection, coins, features, start, end, period)
            """
            for row_number, coin in enumerate(coins):       # There must be a faster way than this double loop
                for feature in features:
                    # NOTE: transform the start date to end date
                    if feature == "close":
                        sql = ("SELECT date+300 AS date_norm, close FROM History WHERE"
                               " date_norm>={start} and date_norm<={end}"
                               " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                               start=start, end=end, period=period, coin=coin))
                    elif feature == "open":
                        sql = ("SELECT date+{period} AS date_norm, open FROM History WHERE"
                               " date_norm>={start} and date_norm<={end}"
                               " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                               start=start, end=end, period=period, coin=coin))
                    elif feature == "volume":
                        sql = ("SELECT date_norm, SUM(volume)"+
                               " FROM (SELECT date+{period}-(date%{period}) "
                               "AS date_norm, volume, coin FROM History)"
                               " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                               " GROUP BY date_norm".format(
                                    period=period,start=start,end=end,coin=coin))
                    elif feature == "high":
                        sql = ("SELECT date_norm, MAX(high)" +
                               " FROM (SELECT date+{period}-(date%{period})"
                               " AS date_norm, high, coin FROM History)"
                               " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                               " GROUP BY date_norm".format(
                                    period=period,start=start,end=end,coin=coin))
                    elif feature == "low":
                        sql = ("SELECT date_norm, MIN(low)" +
                                " FROM (SELECT date+{period}-(date%{period})"
                                " AS date_norm, low, coin FROM History)"
                                " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                                " GROUP BY date_norm".format(
                                    period=period,start=start,end=end,coin=coin))
                    else:
                        msg = ("The feature %s is not supported" % feature)
                        logging.error(msg)
                        raise ValueError(msg)
#                    logging.error('sql command = ' + sql)
                    serial_data = pd.read_sql_query(sql, con=connection,
                                                    parse_dates=["date_norm"],
                                                    index_col="date_norm")
#                    logging.error('serial_data = ' + str(serial_data)) # tuple
#                    logging.error('squeezed serial_data = ' + str(serial_data.squeeze()))
                    panel.loc[feature, coin, serial_data.index] = serial_data.squeeze()
                    panel = panel_fillna(panel, "both")
            """
        finally:
            connection.commit()
            connection.close()
            logging.error("get_global_panel double loop done after " + str(int(time.time() - start_ts)) + " seconds.")
        return panel

    def __get_data_augmented (self, panel, connection, coins, features, start, end, period):
        for row_number, coin in enumerate(coins):       # There must be a faster way than this double loop
            for feature in features:
                # NOTE: transform the start date to end date
                if feature == "close":
                    sql = ("SELECT date+{storage_period} AS date_norm, close FROM History WHERE"
                           " date_norm>={start} and date_norm<={end}"
#                           " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                           " and coin=\"{coin}\"".format(
                               storage_period=self.__storage_period,start=start, end=end, period=period, coin=coin))
                elif feature == "open":
                    sql = ("SELECT date+{period} AS date_norm, open FROM History WHERE"
                           " date_norm>={start} and date_norm<={end}"
#                           " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                           " and coin=\"{coin}\"".format(
                               start=start, end=end, period=period, coin=coin))
                elif feature == "high":
#                    sql = ("SELECT date+{period} AS date_norm,"                   # the proposed new expression for high
                    sql = ("SELECT date+{date_offset} as date_norm,"                   # the proposed new expression for high
#                           "       MAX(high) OVER (ORDER BY date ASC ROWS {further_samples} FOLLOWING) AS high," So preceding works but following doesn't hmmm.
                           "       MAX(high) OVER (ORDER BY date ASC ROWS {further_samples} PRECEDING) AS high "
                           "FROM   History "
#                           "WHERE  date_norm>={start} and date_norm<={end} and coin=\"{coin}\" and date_norm%{period}=0".format(
                           "WHERE  date_norm>={start} and date_norm<={end} and coin=\"{coin}\"".format(
                               period=period,date_offset=self.__storage_period,start=start,end=end,#-period+self.__storage_period,
                               coin=coin,further_samples=(period//self.__storage_period - 1)))
                elif feature == "low":
#                    sql = ("SELECT date+{period} AS date_norm,"                   # the proposed new expression for high
#                           "       MIN(low) OVER (ORDER BY date ASC ROWS {further_samples} FOLLOWING) AS low,"
                    sql = ("SELECT date+{date_offset} as date_norm,"                   # the proposed new expression for high
                           "       MIN(low) OVER (ORDER BY date ASC ROWS {further_samples} PRECEDING) AS low "
                           "FROM   History "
#                           "WHERE  date_norm>={start} and date_norm<={end} and coin=\"{coin}\" and date_norm%{period}=0".format(
                           "WHERE  date_norm>={start} and date_norm<={end} and coin=\"{coin}\"".format(
                               period=period,date_offset=self.__storage_period,start=start,end=end,
                               coin=coin,further_samples=(period//self.__storage_period - 1)))
                else:
                    msg = ("The feature %s is not supported" % feature)
                    logging.error(msg)
                    raise ValueError(msg)
#                logging.error("sql = " + sql)
                serial_data = pd.read_sql_query(sql, con=connection,
                                                parse_dates=["date_norm"],
                                                index_col="date_norm")
#                logging.error(coin + " " + feature + " serial_data " + "(shape=" + str(serial_data.shape) + ") = " + str(serial_data))
                panel.loc[feature, coin, serial_data.index] = serial_data.squeeze()
                panel = panel_fillna(panel, "both") # Am I redoing this thing over and over?
        return panel

    def __get_data (self, panel, connection, coins, features, start, end, period):
        for row_number, coin in enumerate(coins):       # There must be a faster way than this double loop
            for feature in features:
                # NOTE: transform the start date to end date
                if feature == "close":
                    sql = ("SELECT date+300 AS date_norm, close FROM History WHERE"
                           " date_norm>={start} and date_norm<={end}"
                           " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                               start=start, end=end, period=period, coin=coin))
                elif feature == "open":
                    sql = ("SELECT date+{period} AS date_norm, open FROM History WHERE"
                           " date_norm>={start} and date_norm<={end}"
                           " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                               start=start, end=end, period=period, coin=coin))
                elif feature == "volume":
                    sql = ("SELECT date_norm, SUM(volume)"+
                           " FROM (SELECT date+{period}-(date%{period}) "
                           "AS date_norm, volume, coin FROM History)"
                           " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                           " GROUP BY date_norm".format(
                                period=period,start=start,end=end,coin=coin))
                elif feature == "high":
                    sql = ("SELECT date_norm, MAX(high)" +
                           " FROM (SELECT date+{period}-(date%{period})"
                           " AS date_norm, high, coin FROM History)"
                           " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                           " GROUP BY date_norm".format(
                                period=period,start=start,end=end,coin=coin))
                elif feature == "low":
                    sql = ("SELECT date_norm, MIN(low)" +
                           " FROM (SELECT date+{period}-(date%{period})"
                           " AS date_norm, low, coin FROM History)"
                           " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                           " GROUP BY date_norm".format(
                                period=period,start=start,end=end,coin=coin))
                else:
                    msg = ("The feature %s is not supported" % feature)
                    logging.error(msg)
                    raise ValueError(msg)
                serial_data = pd.read_sql_query(sql, con=connection,
                                                parse_dates=["date_norm"],
                                                index_col="date_norm")
#                logging.error(coin + " " + feature + " serial_data = " + str(serial_data))
                panel.loc[feature, coin, serial_data.index] = serial_data.squeeze()
                panel = panel_fillna(panel, "both") # Am I redoing this thing over and over?
        return panel

    # select top coin_number of coins by volume from start to end
    def select_coins(self, start, end):
        # Cache the coin list on disk. That way we'll get the same one on every run with the same algo.
        if self._net_dir != None:
            coinlist_fn = self._net_dir + "/coinlist.json";
            if (os.path.isfile(coinlist_fn)):
                logging.error("Found coin list at " + coinlist_fn + ". Using that instead of calculating")
                fh = open (coinlist_fn, "r")
                coins = json.load(fh)
                fh.close()
                return coins
            logging.error('Did not find ' + coinlist_fn)
        logging.error("select_coins: self._online=" + str(self._online) + " self._live=" + str(self._live));
#        if (not self._online) or (not self._live): #False (should be and?)
        if (not self._online) or (self._live): #False (should be and?)
#        if False:
            logging.error("select coins offline from %s to %s" % (datetime.utcfromtimestamp(start).strftime('%Y-%m-%d %H:%M'),
                                                                 datetime.utcfromtimestamp(end).strftime('%Y-%m-%d %H:%M')))
            connection = sqlite3.connect(const.DATABASE_DIR + '.' + self.market)
            try:
                cursor = connection.cursor()
                cursor.execute('SELECT coin,SUM(volume) AS total_volume FROM History WHERE'
                               ' date>=? and date<=? GROUP BY coin'
                               ' ORDER BY total_volume DESC LIMIT ?;',
                               (int(start), int(end), self._coin_number))
                coins_tuples = cursor.fetchall()

                if len(coins_tuples) != self._coin_number:
#                    logging.error("sqlite error: len(coin_tuples)=" + str(len(coins_tuples)) + " != self._coin_number=" + str(self._coin_number));
                    assert False, "sqlite error: len(coin_tuples)=" + str(len(coins_tuples)) + " != self._coin_number=" + str(self._coin_number)
            finally:
                connection.commit()
                connection.close()
            coins = []
            for tuple in coins_tuples:
                coins.append(tuple[0])
        else:
            logging.error("Getting offline coin list directly from CoinList (no DB query)")
            coins = list(self._coin_list.topNVolume(n=self._coin_number).index)
        logging.error("Selected coins are: "+str(coins))
        if self._net_dir != None:
            logging.info("Saving coin list to " + coinlist_fn)
            try:
                fh = open (coinlist_fn, "w")
                json.dump(coins, fh)
                fh.close()
            except PermissionError:
                logging.error("Failed to write to " + coinlist_fn);
        return coins

    def __checkperiod(self, period):
        if period == const.FIVE_MINUTE:
            return
        elif period == const.FIFTEEN_MINUTE:
            return
        elif period == const.HALF_HOUR:
            return
        elif period == const.TWO_HOUR:
            return
        elif period == const.FOUR_HOUR:
            return
        elif period == const.DAY:
            return
        else:
            raise ValueError('peroid has to be 5min, 15min, 30min, 2hr, 4hr, or a day')

    # add new history data into the database
    def update_data(self, start, end, coin):
        connection = sqlite3.connect(const.DATABASE_DIR + '.' + self.market)
        try:
            cursor = connection.cursor()
            min_date = cursor.execute('SELECT MIN(date) FROM History WHERE coin=?;', (coin,)).fetchall()[0][0]
            max_date = cursor.execute('SELECT MAX(date) FROM History WHERE coin=?;', (coin,)).fetchall()[0][0]
#            logging.info("update_data: db date range for " + coin + ": [" + str(min_date) + ", " + str(max_date) + "] = [" + datetime.fromtimestamp(min_date).strftime('%Y-%m-%d %H:%M %Z(%z)') + ', ' + datetime.fromtimestamp(max_date).strftime('%Y-%m-%d %H:%M %Z(%z)') + ']')
            logging.error("update_data: db date range for " + coin + ": [" + str(min_date) + ", " + str(max_date) + "]")
            if min_date != None and max_date != None:
                logging.error("update_data: db date range for " + coin + ": [" + str(min_date) + ", " + str(max_date) + "] = [" + datetime.utcfromtimestamp(min_date).strftime('%Y-%m-%d %H:%M %Z(%z)') + ', ' + datetime.utcfromtimestamp(max_date).strftime('%Y-%m-%d %H:%M %Z(%z)') + ']')
            logging.error("update_data: requested date range: [" + str(start) + ", " + str(end) + "]")
            logging.error("update_data: __storage_period: " + str(self.__storage_period))

            if (min_date is None) or (max_date is None):
                logging.error("update_data: DB is empty, fetching full data!")
                self.__fill_data(start, end, coin, cursor)
            else:
#                if max_date+10*self.__storage_period<end:       # What is this fuckery? Assuming end is aligned to period boundary, this should be 1, not 10
                if max_date+self.__storage_period<end:       # What is this fuckery? Assuming end is aligned to period boundary, this should be 1, not 10
                                                             # Now we are downloading again and again?! This needs to be revised. (or maybe 'cause it's first run...)
                    if not self._online:
                        raise Exception("Have to be online")
                    logging.error("update_data: Filling data to the end of " + coin + ": [" + str(max_date+self.__storage_period) + ", " + str(end) + "] = [" + datetime.utcfromtimestamp(max_date+self.__storage_period).strftime('%Y-%m-%d %H:%M %Z(%z)') + ', ' + datetime.utcfromtimestamp(end).strftime('%Y-%m-%d %H:%M %Z(%z)') + ']')
                    self.__fill_data(max_date + self.__storage_period, end, coin, cursor) #
                if min_date > start and self._online:
                    logging.error("update_data: Filling data from the start of " + coin + ": [" + str(start) + ", " + str(min_date - self.__storage_period - 1) + "] = [" + datetime.utcfromtimestamp(start).strftime('%Y-%m-%d %H:%M %Z(%z)') + ', ' + datetime.utcfromtimestamp(min_date - self.__storage_period - 1).strftime('%Y-%m-%d %H:%M %Z(%z)') + ']')
                    self.__fill_data(start, min_date - self.__storage_period - 1, coin, cursor) #

            # if there is no data
        finally:
            connection.commit()
            connection.close()
            logging.error("update_data: done updating " + coin)

    def __fill_data(self, start, end, coin, cursor):
        logging.error("__fill_data: fill %s data from %s to %s" % (coin, datetime.utcfromtimestamp(start).strftime('%Y-%m-%d %H:%M %Z(%z)'),
                                                     datetime.utcfromtimestamp(end).strftime('%Y-%m-%d %H:%M %Z(%z)')))
        chart = self._coin_list.get_chart_until_success(
            pair=self._coin_list.allActiveCoins.at[coin, 'pair'],
            start=start,
            end=end,
            period=self.__storage_period)
        logging.error ('raw chart -- ' + str (chart))
#        logging.info("fill %s data from %s to %s" % (coin, datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M %Z(%z)'),
#                                                     datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M %Z(%z)')))
        for c in chart:
            if c["date"] > 0:
                if c['weightedAverage'] == 0:
                    weightedAverage = c['close']
                else:
                    weightedAverage = c['weightedAverage']

                # NOTE here the USDT is in reversed order
                if 'reversed_' in coin:
                    logging.error('Writing to DB: INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)',(c['date'],coin,1.0/c['low'],1.0/c['high'],1.0/c['open'],1.0/c['close'],c['quoteVolume'],c['volume'],1.0/weightedAverage)) # Not all converted
#                    logging.error('Writing to DB: INSERT INTO History VALUES (?,?,?,?,?,?,?)',(c['date'],coin,1.0/c['low'],1.0/c['high'],1.0/c['open'],1.0/c['close'],c['quoteVolume']))
                    cursor.execute('INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)',
                        (c['date'],coin,1.0/c['low'],1.0/c['high'],1.0/c['open'],
                        1.0/c['close'],c['quoteVolume'],c['volume'],
                        1.0/weightedAverage))
                else:
                    cursor.execute('INSERT INTO History VALUES (?,?,?,?,?,?,?,?,?)',
                                   (c['date'],coin,c['high'],c['low'],c['open'],
                                    c['close'],c['volume'],c['quoteVolume'],
                                    weightedAverage))

    def calculate_consumption_vector (self, config):
        raw_consumption = config['trading']['consumption_vector']
        coin_list = self.__coins
        logging.error('raw_consumption -- ' + str(raw_consumption))
        logging.error('coin_list -- ' + str(coin_list))
        ret = []
        for coin in coin_list:
            if coin == 'reversed_USDT':
                market = 'USDT_BTC'
            else:
                market = 'BTC_' + coin
            consumption = raw_consumption[market] / 10000.
            ret.append(consumption)
        logging.error('consumption vector -- ' + str(ret))
        return ret

