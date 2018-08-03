from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from pgportfolio.marketdata.poloniex import Poloniex
from pgportfolio.marketdata.binance import Binance
from pgportfolio.tools.data import get_chart_until_success
import pandas as pd
from datetime import datetime
import logging
import pgportfolio.constants as const
import os.path
import json

class CoinList(object):
    def __init__(self, market, end, volume_average_days=1, volume_forward=0, live=False, net_dir=""):
        if market == "binance":
            self._market = Binance()
        elif market == "poloniex":
            self._market = Poloniex()
        # connect the internet to accees volumes
        vol = self._market.marketVolume()
        ticker = self._market.marketTicker()
        pairs = []
        coins = []
        volumes = []
        prices = []
        net_dir = net_dir.replace("/netfile", "")
        coins_fn = net_dir + "/coins.json";

#        if live == True and os.path.exists(net_dir) and os.path.isfile(coins_fn): # or if coin list file exists
        if os.path.exists(net_dir) and os.path.isfile(coins_fn): # or if coin list file exists
            logging.error("Fetching coin list from file" + coins_fn)
            fh = open(coins_fn)
#            self._df.read_json(fh)
            self._df = pd.read_json(fh)
            fh.close()
#            self._df = json.load(net_dir + "/coin_list.json")
            logging.error("Got coin list from file: " + self._df.to_json());
            return
        else:
            logging.error("Either not live or coin list doesn't exist at " + coins_fn)
            
#            return contents of file.

        logging.info("select coin online from %s to %s" % (datetime.fromtimestamp(end - (const.DAY * volume_average_days) -
                                                                                  volume_forward).
                                                           strftime('%Y-%m-%d %H:%M'),
                                                           datetime.fromtimestamp(end - volume_forward).
                                                           strftime('%Y-%m-%d %H:%M')))
        for k, v in vol.items():
            if (k.startswith("BTC_") or k.endswith("_BTC")) and k not in self._polo.banlist:
                pairs.append(k)
                for c, val in v.items():
                    if c != 'BTC':
                        if k.endswith('_BTC'):
                            coins.append('reversed_' + c)
                            prices.append(1.0 / float(ticker[k]['last']))
                        else:
                            coins.append(c)
                            prices.append(float(ticker[k]['last']))
                    else:
                        volumes.append(self.__get_total_volume(pair=k, global_end=end,
                                                               days=volume_average_days,
                                                               forward=volume_forward))
        self._df = pd.DataFrame({'coin': coins, 'pair': pairs, 'volume': volumes, 'price': prices})
        self._df = self._df.set_index('coin')
        # Write coin list to file
        if (os.path.exists(net_dir)):
            logging.error("Writing coin list to file " + coins_fn)
            _json = self._df.to_json()
            #json.dump(self._df, net_dir + "/coin_list.json");
            fh = open(coins_fn, "w")
            fh.write(_json)
            fh.close()
        else:
            logging.error("Found no folder at " + net_dir)

    @property
    def allActiveCoins(self):
        return self._df

    @property
    def allCoins(self):
        return self._market.marketStatus().keys()

    @property
    def market(self):
        return self._market

    def getBalances(self):
#        balances = self._polo.balances(_polo);
#        balances = self._polo.balances(self._polo);
        balances = self._market.balances(); # TODO: add to binance
        # Ignore coins not managed by this algo.
#        logging.error("Balances: {}".format(balances));
        return balances;

    def get_chart_until_success(self, pair, start, period, end):
        return get_chart_until_success(self._market, pair, start, period, end)

    # get several days volume
    def __get_total_volume(self, pair, global_end, days, forward):
        start = global_end - (const.DAY * days) - forward
        end = global_end - forward
        chart = self.get_chart_until_success(pair=pair, period=const.DAY, start=start, end=end)
        result = 0
        for one_day in chart:
            if pair.startswith("BTC_"):
                result += one_day['volume']
            else:
                result += one_day["quoteVolume"]
        return result

    def topNVolume(self, n=5, order=True, minVolume=0):
        if minVolume == 0:
            r = self._df.loc[self._df['price'] > 2e-6]
            r = r.sort_values(by='volume', ascending=False)[:n]
            print(r)
            if order:
                return r
            else:
                return r.sort_index()
        else:
            return self._df[self._df.volume >= minVolume]
