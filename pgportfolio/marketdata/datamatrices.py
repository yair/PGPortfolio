from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import pgportfolio.marketdata.globaldatamatrix as gdm
import numpy as np
import pandas as pd
import logging
from pgportfolio.tools.configprocess import parse_time
from pgportfolio.tools.data import get_volume_forward, get_type_list
import pgportfolio.marketdata.replaybuffer as rb
#import dumper
import pprint
import datetime
import traceback

MIN_NUM_PERIOD = 3


class DataMatrices:
    def __init__(self, start, end, period, batch_size=50, volume_average_days=30, buffer_bias_ratio=0,
                 market="poloniex", coin_filter=1, window_size=50, feature_number=3, test_portion=0.15,
                 portion_reversed=False, online=False, is_permed=False, live=False, net_dir="",
                 augment_train_set=False):
        """
        :param start: Unix time
        :param end: Unix time
        :param access_period: the data access period of the input matrix.
        :param trade_period: the trading period of the agent.
        :param global_period: the data access period of the global price matrix.
                              if it is not equal to the access period, there will be inserted observations
        :param coin_filter: number of coins that would be selected
        :param window_size: periods of input data
        :param train_portion: portion of training set
        :param is_permed: if False, the sample inside a mini-batch is in order
        :param validation_portion: portion of cross-validation set
        :param test_portion: portion of test set
        :param portion_reversed: if False, the order to sets are [train, validation, test]
        else the order is [test, validation, train]
        """
        #assert False
        start = int(start)
        self.__start = start
        self.__end = int(end)

        # assert window_size >= MIN_NUM_PERIOD
        self.__augment_train_set = augment_train_set
        self.__coin_no = coin_filter
        logging.error("Number of features (should be 3): " + str(feature_number))
        logging.error("Initializing DataMatrices from " + str(start) + " to " + str(int(end)));
        type_list = get_type_list(feature_number) # TODO: Why do we get the volume not supported warning from here? Should be 3
        self.__features = type_list
        self.feature_number = feature_number
        volume_forward = get_volume_forward(self.__end - start, test_portion, portion_reversed)
        self.__history_manager = gdm.HistoryManager(market=market, coin_number=coin_filter, end=self.__end,
                                                    volume_average_days=volume_average_days,
                                                    volume_forward=volume_forward, online=online,
                                                    live=live, net_dir=net_dir, augment_train_set=augment_train_set)
        if market in ["poloniex", "binance"]:
            self.__global_data = self.__history_manager.get_global_panel(start,
                                                                         self.__end,
                                                                         period=period,
                                                                         features=type_list)
        else:
            raise ValueError("market {} is not valid".format(market))
        self.__period_length = period
        self.aug_factor = self.__history_manager.get_aug_factor (period) # Basically how many storage_period offsets can we fit in a global_period
        # portfolio vector memory, [time, assets]
        self.__PVM = pd.DataFrame(index=self.__global_data.minor_axis,
                                  columns=self.__global_data.major_axis)
        if live:
            self.__PVM = self.__PVM.fillna(dict(zip(self.__global_data.major_axis, [1.0] + [0.0] * coin_filter)));
        else:
            self.__PVM = self.__PVM.fillna(1.0 / self.__coin_no)

        self._window_size = window_size
        self._num_periods = len(self.__global_data.minor_axis)
        self.__divide_data(test_portion, portion_reversed)

        self._portion_reversed = portion_reversed
        self.__is_permed = is_permed

        self.__market = market

        self.__batch_size = batch_size
        self.__delta = 0  # the count of global increased
        end_index = self._train_ind[-1]
        self.__replay_buffer = rb.ReplayBuffer(start_index=self._train_ind[0],
                                               end_index=end_index,
                                               sample_bias=buffer_bias_ratio,
                                               batch_size=self.__batch_size,
                                               coin_number=self.__coin_no,
                                               is_permed=self.__is_permed,
                                               aug_factor=self.aug_factor if self.__augment_train_set else 1)

        logging.error("the number of training examples is %s"
                     ", of test examples is %s" % (self._num_train_samples, self._num_test_samples))
        logging.error("the training set is from %s to %s" % (min(self._train_ind), max(self._train_ind)))
        logging.error("the test set is from %s to %s" % (min(self._test_ind), max(self._test_ind)))

    @property
    def global_weights(self):
        return self.__PVM

    @staticmethod
    def create_from_config(config):
        """main method to create the DataMatrices in this project
        @:param config: config dictionary
        @:return: a DataMatrices object
        """
        config = config.copy()
        input_config = config["input"]
        train_config = config["training"]
        start = parse_time(input_config["start_date"])
        end = parse_time(input_config["end_date"])
        return DataMatrices(start=start,
                            end=end,
                            market=input_config["market"],
                            feature_number=input_config["feature_number"],
                            window_size=input_config["window_size"],
                            online=input_config["online"],
                            period=input_config["global_period"],
                            coin_filter=input_config["coin_number"],
                            is_permed=input_config["is_permed"],
                            buffer_bias_ratio=train_config["buffer_biased"], # Boog - this is used from trading and live as well.
                            batch_size=train_config["batch_size"],
                            volume_average_days=input_config["volume_average_days"],
                            test_portion=input_config["test_portion"],
                            portion_reversed=input_config["portion_reversed"],
                            live=input_config["live"],
                            net_dir=input_config["net_dir"],
                            augment_train_set=input_config["augment_train_set"],
                            )

    @property
    def global_matrix(self):
        return self.__global_data

    @property
    def coin_list(self):
        return self.__history_manager.coins

    @property
    def num_train_samples(self):
        return self._num_train_samples

    @property
    def test_indices(self):
        if self.__augment_train_set:
            aug = self.aug_factor
#            return self._test_ind[:-(self._window_size * 6 + 6):] ???
#            return self._test_ind[:-(self._window_size * 6 + 6):6] # or
#            return self._test_ind[5 - self._test_ind[5] % 6 : -(self._window_size * 6 + 6) : 6] # Align to round global period, backoff and skip
            return self._test_ind[aug - 1 - self._test_ind[aug - 1] % aug : -(self._window_size * aug + aug) : aug] # Align to round global period, backoff and skip
        else:
            return self._test_ind[:-(self._window_size + 1):]

    @property
    def train_indices (self):   # Is this just for non-fast-training?
        if self.__augment_train_set:
            ret = []
            for i in range (self.aug_factor):
#                ret += indexs [i : -6 * self._window_size : 6] # still have w leakage across boundary. Actually prices as well... Need some kind of masking. (Fixed in ReplayBuffer)
                ret += self._train_ind[i : -self.aug_factor * self._window_size : self.aug_factor] # still have w leakage across boundary. Actually prices as well... Need some kind of masking? I think the replay buffer takes care of that.
        else:
            ret = self._train_ind [: -self._window_size]
#        logging.error('train_indices: ' + str(ret)) # this isn't chopped, printing out all the numbers, dunno why.
        return ret

    @property
    def num_test_samples(self):
        return self._num_test_samples

    def append_experience(self, online_w=None):
        """
        :param online_w: (number of assets + 1, ) numpy array
        Let it be None if in the backtest case.
        """
        self.__delta += 1
        self._train_ind.append(self._train_ind[-1] + 1)
        appended_index = self._train_ind[-1]
        self.__replay_buffer.append_experience(appended_index)

    def get_test_set(self):
        ret = self.__pack_samples(self.test_indices, live=False) #, skip_augmentation=True)
        logging.error('Our test indices (aug=' + str(self.__augment_train_set) + '): ' + str(self.test_indices))
#        logging.error('This is our test set: ' + str(ret))
        return ret

    def get_training_set(self):
#        return self.__pack_samples(self._train_ind[:-self._window_size], live=False, skip_augmentation=False)
        return self.__pack_samples(self.train_indices, live=False)# , skip_augmentation=False)

    def get_live_set(self, time):
        if self.__market == "poloniex":
            # Why is setting the time range we want wrong? Is this what fooked our results then?
            # And how are we to change the config? If at init time it will become obsolete, and if on every cycle it will contradict members initted from it before.
            # Maybe change it here and find out what caused the bug, if any?
            self.__global_data = self.__history_manager.get_global_panel(self.__start, # time - window * perod?
#            self.__global_data = self.__history_manager.get_global_panel(time - 3 * self._window_size * self.__period_length, # TODO: This is wrong. Change the config
#                                                                         self.__end,
                                                                         time,
                                                                         period=self.__period_length,
#                                                                         features=type_list)
                                                                         features=self.__features)
            self._num_periods = len(self.__global_data.minor_axis)
            self.__PVM = pd.DataFrame(index=self.__global_data.minor_axis,
                                      columns=self.__global_data.major_axis)
            self.__PVM = self.__PVM.fillna(1.0 / self.__coin_no)
#            indexs = np.arange(self._num_periods - 2 * self._window_size - 3, self._num_periods) # cannot end after self._num_periods - self._window_size - 1 or so
            indexs = np.arange(self._num_periods - self._window_size - 1, self._num_periods - self._window_size)
#            indexs = (self._num_periods - self._window_size - 1)    # is this really the latest we can get? Will its last row be cut off as 'y'? Also, can't run a one element array :p
#            indexs = np.arange(self._num_periods - 1, self._num_periods)
            logging.error('Live indexs: ' + str(indexs) + ' Num periods: ' + str(self._num_periods));
            ret = self.__pack_samples(indexs, live=True) #, skip_augmentation=True) # This is actually an interesting question, but ain't they all
            logging.error('Live set: ' + str(ret))
            return ret
#           return self.__pack_samples(self.test_indices)
#            panel = self.__history_manager.get_global_panel(time - self._window_size * self.__period_length, #start,
#                                                            time, #self.__end,
#                                                            period=self.__period_length,
#                                                            features=self.__features) #type_list)
        else:
            raise ValueError("market {} is not valid".format(market))

    def get_current_balances(self):
        return self.__history_manager.get_current_balances() # We should memorize these, perhaps

    def next_batch(self):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """
        batch = self.__pack_samples([exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    """
    def __expand_indices (self, indexs, skip_augmentation): # TODO: calc consts from params

#        skip_augmentation = True # Test identicality with unaugmented data
        if self.__augment_train_set:
            if skip_augmentation:
                # indexs[0]%6   offset
                # 0             0               6 - % would be correct except for 0 -> 6
                # 1             5               5 - i[1]%6 gives 0->4.
                # 2             4               5 - i[5]%6 gives 0->0 1->5 5->1... ok!
                # 5             1
                ret = indexs[5 - indexs[5] % 6::6] # Right?
                # Prune modulu non-zero samples
            else:
                # Rearrange by modulu, remove samples with broken histories (but the data for that is in get_submatrix!)
                ret = []
                for i in range (6):
                    ret += indexs [i::6]
        else:
            ret = indexs
#        logging.error ('__expand_indices (aug=' + str (self.__augment_train_set) + ', skip=' + str (skip_augmentation) + '): ' + str (ret))
#        logging.error ('__expand_indices data corresponding to first index: ' + str (self.__global_data.values[:, :, ret[0]]))
#        logging.error ('__expand_indices: First sse seems to be ' + str (self.__start + ret[0] * 300) + ' (' + str (datetime.datetime.fromtimestamp(self.__start + ret[0] * 300).strftime('%Y-%m-%d %H:%M:%S')) + ')')
        return ret
    """

    def __pack_samples(self, indexs, live=False): #, skip_augmentation=True):
#        logging.error("first line of global data: " + str(self.__global_data.values[:, :,0]))
#        process.exit(1)
#        indexs = np.array(self.__expand_indices (indexs, skip_augmentation))                   # 1D numpy array (ndarray)
        indexs = np.array (indexs)                   # 1D numpy array (ndarray)
#        logging.error("indexs = " + str(indexs))
        if live:
            logging.error("\n\n__pack_samples: indexs type is {}".format(type(indexs).__name__) + ' and its shape is {}'.format(indexs.shape))
            logging.error("indexs = " + str(indexs))
#        else:
#            logging.error("WTF?! We're in __pack_samples but not alive. indexs.shape=" + str(indexs.shape))
#            traceback.print_stack()

        # What happens if we take last_w from history just some of time, and set it to (1,0...0) or (1/n,1/n,...) some of the times, to encourage exploration?
        # last_w = self.__PVM.values[indexs - 1, :] # Shape is [batch_size, noof_coins]  BOOG - that -1 is broadcast!
        last_w = self.__PVM.values[indexs - self.aug_factor, :] # Shape is [batch_size, noof_coins]
#        logging.error('__pack_samples: w indices are [' + str(indexs[0]) + ', ' + str(indexs[-1]) + '] last_w indices are [' + str((indexs-self.aug_factor)[0]) + ', ' + str((indexs-self.aug_factor)[-1]) + ']')
        if live:
            logging.error("last_w (from indexs-1 = " + str(indexs-1) + ") = " + str(last_w))
#        last_w = [0.5 * (self.__PVM.values[index-1] + np.softmax(np.random(self.__PVM.shape[1])) for index in indexs] # invalid syntax
#        logging.error("__pack_samples: last_w shape is {}".format(type(last_w).__name__) + ' and its shape is {}'.format(last_w.shape))

        def setw(w):
            if live:
                logging.error('Live setting new w=' + str(w) + ' to indexs ' + str(indexs))
            self.__PVM.iloc[indexs, :] = w
#        for index in indexs:
#            logging.error('submatrix for index + ' + str(index) + ': ' + str(self.get_submatrix(index)))
        M = [self.get_submatrix(index) for index in indexs] # <--- the only list (and only non-ndarray)
#        if ("{}".format(indexs.shape) == "(1,)"):
#            dumper.dump(M)
        if live:
            logging.error("__pack_samples: M's type is " + type(M).__name__ + ". Its length is {}".format(len(M)) + ". record type is " + type(M[0]).__name__ + " record shape is {}".format(M[0].shape))
#        else:
#            logging.error("WTF?! We're in __pack_samples but not alive. M's type is " + type(M).__name__ + ". Its length is {}".format(len(M)) + ". record type is " + type(M[0]).__name__ + " record shape is {}".format(M[0].shape))
#        for i in range (0, 4):
#            logging.error("__pack_samples: M[{}] shape is ".format(i) + type(M[i]).__name__ + " and its shape is {}".format(M[i].shape))
#        logging.error("__pack_samples: M[0][0][0][0] is of type {}".format(type(M[0][0][0][0]).__name__))
#        for i in range (0, len(M)):
#            if ("{}".format(M[i].shape) != "(3, 5, 32)"):
#                logging.error("\nOops: shape of M[{}] is ".format(i) + "{}".format(M[i].shape))
#        logging.error("\n\nM before --")
#        dumper.dump(M);
        M = np.array(M)
#        logging.error("__pack_samples: And now M's type is " + type(M).__name__ + " and its shape is {}".format(M.shape))# + ". Its length is {}".format(len(M)) + ". record type is " + M[0].shape())
#        logging.error("\n\nM after --")
#        dumper.dump(M);
#        logging.error("Post-conversion M shape is {}.\n\n".format(M.shape()))
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]
        if (live):
            X = M[:, :, :, 1:]
            y = np.zeros(y.shape)
        else:
            X = M[:, :, :, :-1]
#        logging.error("__pack_samples: X type is {}".format(type(X).__name__) + " and its shape is {}".format(X.shape))
#        logging.error("__pack_samples: X=" + pprint.pformat(X));
#        logging.error("__pack_samples: Y type is {}".format(type(y).__name__) + " and its shape is {}".format(y.shape))
#        logging.error("__pack_samples: y=" + pprint.pformat(y));
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    # volume in y is the volume in next access period
    def get_submatrix(self, ind):
        if self.__augment_train_set:
#            logging.error ('augmented get_submatrix getting global_data.values[:, :, ' + str(ind) + ':' + str(ind + (self._window_size + 1) * self.aug_factor) + ':' + str(self.aug_factor) + ']')
            # logging.error ('augmented get_submatrix ind = ' + str (ind) + ' and ind + (self._window_size + 1) * 6 = ' + str (ind + (self._window_size + 1) * 6) + '. Output shape is ' + str (self.__global_data.values[:, :, ind:ind + (self._window_size + 1) * 6:6].shape))
            # ret = indexs[5 - indexs[5] % 6::6] # Right?
            # ret = self.__global_data.values[:, :, ind:ind + (self._window_size + 1) * 6:6]
            ret = self.__global_data.values[:, :, ind:ind + (self._window_size + 1) * self.aug_factor:self.aug_factor]
        else:
#            logging.error ('non-augmented get_submatrix getting global_data.values[:, :, ' + str(ind) + ':' + str(ind + self._window_size + 1) + ']')
            ret = self.__global_data.values[:, :, ind:ind + self._window_size + 1]
        return ret # self.__global_data.values[:, :, ind:ind + self._window_size + 1]

    def __divide_data(self, test_portion, portion_reversed):
        train_portion = 1 - test_portion
        s = float(train_portion + test_portion)
        if portion_reversed:
            portions = np.array([test_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._test_ind, self._train_ind = np.split(indices, portion_split)
        else:
            portions = np.array([train_portion]) / s
            portion_split = (portions * self._num_periods).astype(int)
            indices = np.arange(self._num_periods)
            self._train_ind, self._test_ind = np.split(indices, portion_split)

        # Yair - this was a bug. Backing off the test set was also implemented in get_train_set (and anyway moved to train_indices with new augmentation code)
        # self._train_ind = self._train_ind[:-(self._window_size + 1)]
        # NOTE(zhengyao): change the logic here in order to fit both
        # reversed and normal version
        self._train_ind = list(self._train_ind)
        self._num_train_samples = len(self._train_ind)
        self._num_test_samples = len(self.test_indices)

    def calculate_consumption_vector (self, config):
        return self.__history_manager.calculate_consumption_vector (config)
