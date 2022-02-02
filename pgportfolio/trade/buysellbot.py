from __future__ import absolute_import, division, print_function

# Yes. No. Just wrap the whole thing in a nice json and ship it off to node. Nicer libs, nicer language. Buhbye. Or not. Grrr

import numpy as np
import logging
import pprint
#from poloniex import Poloniex as pololib
from threading import Thread
from time import sleep
import json
#from autobahn.twisted.websocket import WebSocketClientProtocol, WebSocketClientFactory
#from twisted.internet import reactor
import re
import time
import sys
#from twisted.python import log
import os
import inotify.adapters
from subprocess import call

PAPER_TRADE = True
MINIMUM_TRADE = 0.0001          #10ksat @poloni
PRICE_RESOLUTION = 0.00000001   #1sat @poloni
"""
class WSSHandler (WebSocketClientProtocol):
    def __init__(self, bst):
        self._bst = bst
        logging.error("WSS handler is alive");
    def onPing(self):
        logging.error("Ping")
    def onPong(self):
        logging.error("Pong")
    def onOpen(self):
        logging.error("Opened a wss connection");
        self.sendMessage(u"Hello, world!".encode('utf8'))
    def onMessage(self, payload, isBinary):
      if isBinary:
         print("Binary message received: {0} bytes".format(len(payload)))
      else:
         print("Text message received: {0}".format(payload.decode('utf8')))

class BuySellThread (Thread):
    def __init__(self, action, timeout):
        Thread.__init__(self)
        self._action = action
        self._strategies = [ lambda x=0: self.doubleup(x, .5),
                             lambda x=0: self.doubleup(x, 1.),
                             lambda x=0: self.doubleup(x, 2.),
                             lambda x=0: self.bottom(x),
                             lambda x=0: self.midway(x),
                             lambda x=0: self.top(x),
                             lambda x=0: self.market_buy(x) ]
        self._timeout = timeout
        m = re.match('reversed_(\w+)$', action[1])                          # Should use the transformer if revived
        if (m != None):
            self._coin = m.group(1)
            self._market = "BTC" + m.group(1)
            self._side = "Sell" if action[0] == "Buy" else "Sell"
        else:
            self._coin = action[1]
            self._market = action[1] + "BTC"
            self._side = action[0]
        self._orig_balance = action[2]
        self._cur_balance = action[2]
        self._orig_amount = action[3]
#        self._wss_handler = WSSHandler(self)

    def doubleup(self, amount, ratio):
        logging.error("Current strategy: double up with ratio of {}.".format(ratio))
        return []

    def top(self, amount):
        logging.error("Current strategy: top")
        return []

    def midway(self, amount):
        logging.error("Current strategy: midway")
        return []

    def bottom(self, amount):
        logging.error("Current strategy: bottom")
        return []

    def market_buy(self, amount):
        logging.error("Current strategy: market buy")
        return [(amount, float("inf"))]

    def compare_orders(self, new_orders):
        return new_orders

    def execute_delta(self):
        for order in self._order_delta:
            if order[1] == float("inf"):
                return True
        return False

    def run(self):
        logging.error("Hello there from the {}".format(self._action[1]) + " {} thread! ".format(self._action[0]) +
                      "Now buying {} coins".format(self._orig_amount) + " in {}s.".format(self._timeout))

        start= int(time.time())
        # create an syncio lock
        log.startLogging(sys.stdout)
        factory = WebSocketClientFactory(proxy={'host': 'socks5://127.0.0.1', 'port': 4711})
        factory.protocol = WSSHandler(self)
        logging.error("Before connect")
        reactor.connectTCP("wss://api2.poloniex.com/", 80, factory)
        logging.error("After connect")
        reactor.run(installSignalHandlers=0)
        logging.error("After run")

        # register to WAMP async push interface for trades and orderbook changes (do we get double notified?)

        # load orderbook
        # We need some (~20?) on our side, and one on the other side

        self._update_orders = 1
        self._refetch_balance = 0
        self._orders = []
        self._order_delta = []  # must be member to filter out in WAMP handler
        self._order_book = []
        self._opp_order = []

        while True:
            stage = int((time.time() - start) * (len(self._strategies) - 1) / self._timeout)   # last one is always a market order, which we assume takes no time
            strategy = self._strategies[stage]
            if self._refetch_balance:
                self._cur_balance = self.get_balance()
            self._refetch_balance = 0
            amount = self._orig_balance + self._orig_amount - self._cur_balance
            if self._update_orders:
                new_orders = strategy(amount)
            self._order_delta = self.compare_orders(new_orders)
            if (self.execute_delta()):    # Ehm, so where do return?
                return
            sleep(1.)       # Enough time for the dust to settle? Can we make it shorter? Can we know that we received all notifications of our own actions and just continue?
"""
            

class BuySellBot:
    def __init__(self, period, coin_list):
        self._period = period
        self._proxies = {'http': 'socks5://127.0.0.1:4711', # socks5://<usr>:<pwd>@<addr>:<port>
                         'https': 'socks5://127.0.0.1:4711'} #use only if you are using **socks**
#        self.polo = pololib(proxies=self._proxies)
#        self.polo = pololib()
        self._coin_list = coin_list
        logging.error("Write coin list to broker volatile dir");

    def coin_name(self, coin):
        return self._coin_list[coin - 1]

    def rebalance_portfolio(self, prev_omega, next_omega, prev_balances, total_capital, prices):
        # Sell
        actions = []
        for coin in range(1, len(prev_omega)):
            if next_omega[coin] < prev_omega[coin]:
#                if new_omega[coin] * total_capital < MINIMUM_TRADE:
                if next_omega[coin] * total_capital < MINIMUM_TRADE: # I think that was the meaning
                    actions.append(("Sell", self._coin_list[coin - 1], prev_balances[coin], prev_balances[coin], prices[coin])) # Dump all. Need to recalc next_omega to boost others!
                else:
                    actions.append(("Sell", self._coin_list[coin - 1], prev_balances[coin], prev_balances[coin] * (1 - (next_omega[coin] / prev_omega[coin])), prices[coin]))

#        self.launch_actions_via_scp(sell_actions, 60) # 300

        # remeasure balance and omega

        # Buy
        for coin in range(1, len(prev_omega)):
            if next_omega[coin] * total_capital > prev_omega[coin] * total_capital + MINIMUM_TRADE:
                actions.append(("Buy", self._coin_list[coin - 1], prev_balances[coin], (next_omega[coin] - prev_omega[coin]) * total_capital / prices[coin], prices[coin]))
#            elif next_omega[coin] * total_capital < prev_omega[coin] * total_capital - MINIMUM_TRADE:
#                actions.append(("Sell", self._coin_list[coin - 1], prev_balances[coin], (prev_omega[coin] - next_omega[coin]) * total_capital / prices[coin], prices[coin]))

        self.launch_actions_via_scp(actions, 360, prev_balances, next_omega) # 300)

        # remeasure balance and omega, and return them

    def launch_actions_via_scp(self, actions, timeout, prev_balances, omega):
        folder = '/home/yair/w/volatile/'
        orders_fn = 'orders.json'
#        results_fn = 'results.json'
#        lock_suffix = '.lock'
#        with open(folder + orders_fn + lock_suffix, 'w') as f:
#            f.close()

        dict_actions = self.transform_actions (actions, prev_balances);

        with open(folder + orders_fn, 'w') as f:
#            - Translate coin names in actions
#            - Add wanted price for reference (in actions)
#            - scp file to dm2
            f.write(json.dumps({'timeout': timeout, 'actions': dict_actions, 'omega': omega.tolist()}))
            f.close()
            logging.error("cmd: scp args: " + folder + orders_fn + " yair@dm2:w/volatile/");
            call(["scp", folder + orders_fn, "yair@dm2:w/volatile/"])
            logging.error("\n---> orders file copied to dm2\n");


#        os.remove(folder + orders_fn + lock_suffix);
        """
        i = inotify.adapters.Inotify()
        i.add_watch(folder)
        for event in i.event_gen(yield_nones=False):
            (_, type_names, path, filename) = event
            print("PATH=[{}] FILENAME=[{}] EVENT_TYPES={}".format(path, filename, type_names))
            if (path == folder) and \
               (filename == results_fn + lock_suffix) and \
               type_names == ['IN_DELETE']:
                logging.error("Results lock removed, moving on.")
                break
        """

    # action[0] - type
    # action[1] - mname
    # action[2] - previous (current) balance
    # action[3] - action amount
    # action[4] - price
    def transform_actions(self, actions, prev_balances):

        ret = []
        for action in actions:
            logging.error("Action before: " + json.dumps(action))
            da = {}
            # What is the reversal algo? Which should be reversed an how?
            m = re.match('reversed_(\w+)$', action[1])                          # Should use the transformer if revived
            if (m != None):     # USDT_BTC
                da['type'] = "Sell" if action[0] == "Buy" else "Sell"
                da['mname'] = m.group(1) + "_BTC" # Used to be "BTC" + m.group(1)
                da['previous_balance'] = prev_balances[0]; # BTC was: action[2] * action[4]
                da['amount'] = action[3] * action[4]
                """
                action[0] = "Sell" if action[0] == "Buy" else "Sell"
                action[1] = m.group(1) + "_BTC" # Used to be "BTC" + m.group(1)
                action[2] = action[2] * action[4]
                action[3] = action[3] * action[4]
                """
            else:
                da['type'] = action[0]
                da['mname'] = "BTC_" + action[1] # used to be coin + "_BTC"
                da['previous_balance'] = action[2]
                da['amount'] = action[3]
            da['price'] = action[4]
            logging.error("Action after: " + json.dumps(da))
            ret.append(da)
        return ret
        """
        threads = []
        for action in actions:
            threads.append(BuySellThread (action, timeout, ))
            threads[-1].start()

        for t in threads:
            t.join() 
        """


