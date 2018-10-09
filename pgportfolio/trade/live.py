from __future__ import absolute_import, division, print_function
import numpy as np
from pgportfolio.trade import trader
from pgportfolio.trade.buysellbot import BuySellBot
from pgportfolio.marketdata.datamatrices import DataMatrices
#from pgportfolio.marketdata.globaldatamatrix import HistoryManager as gdm
import logging
from pgportfolio.tools.trade import calculate_pv_after_commission
import time
#import dumper
import pprint

class LiveTrader(trader.Trader):
    def __init__(self, config, net_dir=None, agent=None, agent_type="nn"):
#        config["input"]["live"] = True  #Kludgy way to propagate the flag, yeah
        config["input"]["net_dir"] = net_dir    # Even kloodjieer, yah
#        config["input"]["start_data"] = "2018/06/01" # Both kludgy and doesn't work
#        config["input"]["end_data"] = "2018/06/10"
        self.__period = config["input"]["global_period"];
        trader.Trader.__init__(self, self.__period, config, 5000, net_dir,
                               initial_BTC=1, agent=agent, agent_type=agent_type)
#        if agent_type == "nn":
#            data_matrices = self._rolling_trainer.data_matrices
#        elif agent_type == "traditional":
#            config["input"]["feature_number"] = 1
#            data_matrices = DataMatrices.create_from_config(config)
#        else:
#            raise ValueError()
#        self.__test_set = data_matrices.get_test_set()                 We want the unsplit global matrices
#        self.__test_length = self.__test_set["X"].shape[0]
#        self._total_steps = self.__test_length
#        self.__test_pv = 1.0
        self.__test_pc_vector = []         # What is this? Do we need it?
        self.__window_size = config["input"]["window_size"];

        logging.error("\nBeware! Beware! Live Trading Starting Now!!1!\n");
        self.__data_matrices = DataMatrices.create_from_config(config)
        self._buysellbot = BuySellBot(self.__period, self.__data_matrices.coin_list)
        self._live_set = None
        self._live_time_stamp = 0

#    @property
#    def test_pv(self):
#        return self.__test_pv

    @property
    def test_pc_vector(self):
        return np.array(self.__test_pc_vector, dtype=np.float32)

    def finish_trading(self):
        self.__test_pv = self._total_capital

        """
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(self._rolling_trainer.data_matrices.sample_count)),
               self._rolling_trainer.data_matrices.sample_count)
        fig.tight_layout()
        plt.show()
        """

    def _log_trading_info(self, time, omega):
        pass

    def _initialize_data_base(self):
        pass

    def _write_into_database(self):
        pass

    def measure_current_omega(self): #, balances, prices):
        # Fetch balances from exchange, multiply by prices, normalize to sum=1.
        self._balances = self.__data_matrices.get_current_balances()
        logging.error("Got balances: " + pprint.pformat(self._balances))
        self._prices = self.__get_matrix_Y()
        logging.error("Got prices: " + pprint.pformat(self._prices))
        self._values = self._balances * self._prices
        logging.error("calced values: " + pprint.pformat(self._values))
        self._total_capital = np.sum(self._values)
        logging.error("Total capital: " + pprint.pformat(self._total_capital))
        self._last_omega = self._values / self._total_capital
        logging.error("New omega: " + pprint.pformat(self._last_omega))
        return self._last_omega

    def __live_set(self):
        self.__now = int(time.time());
        logging.error("now is {}".format(self.__now) + " and period is {}".format(self.__period))
        self.__now = self.__now - (self.__now % self.__period);
        logging.error("Last period ended at {}".format(self.__now));
        logging.error("window size is {}".format(self.__window_size));
        if self.__now != self._live_time_stamp:
            self._live_time_stamp = self.__now
            self._live_set = self.__data_matrices.get_live_set(self.__now)
        return self._live_set

    def __get_matrix_X(self):
        """
#        return self.__test_set["X"][self._steps]
        # Go to exchange, bring a fresh batch.
        self.__now = int(time.time());
        logging.error("now is {}".format(self.__now) + " and period is {}".format(self.__period))
        self.__now = self.__now - (self.__now % self.__period);
        logging.error("Last period ended at {}".format(self.__now));
        logging.error("window size is {}".format(self.__window_size));
#        gdm.get_global_data_matrix(self, self.__now - config["input"]["global_period"] * config["input"]["window_size"], self.__now,
#                                   config["input"]["global_period"], ("close", 'high', 'low'));
#        gdm.get_global_data_matrix(gdm, self.__now - self.__period * self.__window_size, self.__now,
#                                   self.__period, ("close", 'high', 'low'));
"""
#        return self.__data_matrices.get_live_set(self.__now)["X"][0]
        return self.__live_set()["X"][0]
        
    def __get_matrix_Y(self):
        self.__now = int(time.time());
        self.__now = self.__now - (self.__now % self.__period);
        logging.error("Dumping current prices:");
#        live_set = self.__data_matrices.get_live_set(self.__now)
        live_set = self.__live_set()
#        prices = np.concatenate((np.ones(1), live_set["X"][0][0][:,0]))
        prices = np.concatenate((np.ones(1), live_set["X"][0][0][:,-1]))
#        dumper.dump(live_set["X"][0][0][:,0])
        logging.error(pprint.pformat(prices))
        logging.error("\nThat was the live set")
#        return live_set["X"][0][0][:,0]
        return prices
        #self.__test_pc_vector.append(portfolio_change)

    def rolling_train(self, online_sample=None):
        self._rolling_trainer.rolling_train()

    def generate_history_matrix(self):
        inputs = self.__get_matrix_X()
        if self._agent_type == "traditional":
            inputs = np.concatenate([np.ones([1, 1, inputs.shape[2]]), inputs], axis=1)
            inputs = inputs[:, :, 1:] / inputs[:, :, :-1]
#        logging.error("history matrix: " + pprint.pformat(inputs))
        return inputs

    def trade_by_strategy(self, omega):
        logging.info("the step is {}".format(self._steps))
        logging.debug("the raw new omega is {}".format(omega))

        self._buysellbot.rebalance_portfolio(self._last_omega, omega, self._balances, self._total_capital, self._prices)

        # Get Current prices
#        future_price = np.concatenate((np.ones(1), self.__get_matrix_y(self.__now)))
#        logging.info("future_price = {}".format(future_price));
        # Get balances
#        balances = gdm.get_current_balances()
#        logging.info("balances: {}".format(balances));
        # Calc last omega from balances
#        omega = measure_current_omega(self, balances, future_price);
#        logging.info("Omega: {}".format(omega));
        # Generate sell and buy instructions
#        sales, buys =
        # Calculate the predicted effect of the instructions
        # execute them if non-paper-trading

        # Emit stats and prepare stuff for next iteration
        #self._last_omega = omega;

        # This is the backtest stuff. Replace it with actual trading instructions / paper trading

        #pv_after_commission = calculate_pv_after_commission(omega, self._last_omega, self._commission_rate)
        #portfolio_change = pv_after_commission * np.dot(omega, future_price)
        #self._total_capital *= portfolio_change
        #self._last_omega = pv_after_commission * omega * \
        #                   future_price /\
        #                   portfolio_change
        #logging.debug("the portfolio change this period is : {}".format(portfolio_change))
         #self.__test_pc_vector.append(portfolio_change)

