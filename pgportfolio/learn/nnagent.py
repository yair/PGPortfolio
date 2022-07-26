from __future__ import absolute_import, print_function, division
import tflearn
import tensorflow as tf
import numpy as np
from pgportfolio.constants import *
import pgportfolio.learn.network as network
import logging
import pprint

class NNAgent:
    def __init__(self, config, consumption_vector, restore_dir=None, device="cpu"):
        self.__config = config
        self.__coin_number = config["input"]["coin_number"]
        self.set_consumption_vector (consumption_vector)
        self.__net = network.CNN(config["input"]["feature_number"],     # Here we import ../network.py
                                 self.__coin_number,
                                 config["input"]["window_size"],
                                 config["layers"],
                                 device=device,
                                 consumption_vector=consumption_vector,
                                 config=self.__config)
        self.__global_step = tf.Variable(0, trainable=False)
#        self.__train_operation = None                                  # Initialized later
        self.__y = tf.placeholder(tf.float32, shape=[None,
                                                     self.__config["input"]["feature_number"],
                                                     self.__coin_number])
        self.__future_price = tf.concat([tf.ones([self.__net.input_num, 1]),
                                       self.__y[:, 0, :]], 1)
        # I don't understand this. What does prices have to do with this? Omega is the fraction held in each coin. The network output is that fraction divided by price?
        # I think this is to evolve the old omega according to the change in price during the period.
        # Answer - omega changes simply with asset price variation over the period, without buying and selling.
        self.__future_omega = (self.__future_price * self.__net.output) /\
                              tf.reduce_sum(self.__future_price * self.__net.output, axis=1)[:, None]
        logging.error('__net.output.shape is ' + str(self.__net.output.get_shape()))
        logging.error('__future_price.shape is ' + str(self.__future_price.get_shape()))
        # tf.assert_equal(tf.reduce_sum(self.__future_omega, axis=1), tf.constant(1.0))
#        self.__commission_ratio = self.__config["trading"]["trading_consumption"]
#        self.__commission_vector = self.get_commission_vector ()
        # Unnormalized future omega (i.e. total portfolio value) multiplied by speicific trading costs? So real cost?
        # This is what gets fooked up when we switch from constant to by-market costs. But why? Are we getting negative costs?
        self.__pv_vector = tf.reduce_sum(self.__net.output * self.__future_price, reduction_indices=[1]) *\
                           (tf.concat([tf.ones(1), self.__pure_pc()], axis=0))
        # Logarithm of the same without the cost?
        self.__log_mean_free = tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output * self.__future_price,
                                                                   reduction_indices=[1])))
        self.__portfolio_value = tf.reduce_prod(self.__pv_vector) # Why product?
        self.__mean = tf.reduce_mean(self.__pv_vector)
        self.__log_mean = tf.reduce_mean(tf.log(self.__pv_vector))
        self.__standard_deviation = tf.sqrt(tf.reduce_mean((self.__pv_vector - self.__mean) ** 2))
        self.__sharp_ratio = (self.__mean - 1) / self.__standard_deviation
        self.__loss = self.__set_loss_function()
        self.__train_operation = self.init_train(learning_rate=self.__config["training"]["learning_rate"],
                                                 decay_steps=self.__config["training"]["decay_steps"],
                                                 decay_rate=self.__config["training"]["decay_rate"],
                                                 training_method=self.__config["training"]["training_method"])
        self.__saver = tf.train.Saver()
        if restore_dir:
            logging.error("Saved model restore dir found at " + restore_dir + ". Restoring.")
            self.__saver.restore(self.__net.session, restore_dir)
        else:
            logging.error("No saved model restore dir found. Running session.")
            self.__net.session.run(tf.global_variables_initializer())

    @property
    def session(self):
        return self.__net.session

    @property
    def pv_vector(self):
        return self.__pv_vector

    @property
    def standard_deviation(self):
        return self.__standard_deviation

    @property
    def portfolio_weights(self):
        return self.__net.output

    @property
    def sharp_ratio(self):
        return self.__sharp_ratio

    @property
    def log_mean(self):
        return self.__log_mean

    @property
    def log_mean_free(self):
        return self.__log_mean_free

    @property
    def portfolio_value(self):
        return self.__portfolio_value

    @property
    def loss(self):
        return self.__loss

    @property
    def layers_dict(self):
        return self.__net.layers_dict

    def recycle(self):
        logging.error("\n\n\nResetting graph and closing session!\n\n\n")
        tf.reset_default_graph()
        self.__net.session.close()

    def __entropy (self):
        pass
#        self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
#        self.losses = - (tf.log(self.picked_action_probs) * self.targets + 0.01 * self.entropy)
#        self.loss = tf.reduce_sum(self.losses, name="loss")

    def __set_loss_function(self):
        def loss_function4():
            return -tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output[:] * self.__future_price,
                                                        reduction_indices=[1])))

        def loss_function5():
            return -tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output * self.__future_price, reduction_indices=[1]))) + \
                   LAMBDA * tf.reduce_mean(tf.reduce_sum(-tf.log(1 + 1e-6 - self.__net.output), reduction_indices=[1]))

        def loss_function6():
            return -tf.reduce_mean(tf.log(self.pv_vector))      # <--- we use this

        def loss_function7():
            return -tf.reduce_mean(tf.log(self.pv_vector)) + \
                   LAMBDA * tf.reduce_mean(tf.reduce_sum(-tf.log(1 + 1e-6 - self.__net.output), reduction_indices=[1]))

        def with_last_w():
            assert False
            return -tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output[:] * self.__future_price, reduction_indices=[1])
                                          -tf.reduce_sum(tf.abs(self.__net.output[:, 1:] - self.__net.previous_w)
                                                         *0.0025, reduction_indices=[1]))) # Too optimistic, shouldn't be used.
#                                                         *self.__commission_ratio, reduction_indices=[1])))

        def with_last_w_cv():
            assert False
            cv = self.consumption_vector
            return -tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output[:] * self.__future_price, reduction_indices=[1])
                                          -tf.reduce_sum(tf.matmul(tf.abs(self.__net.output[:, 1:] - self.__net.previous_w), 2 * cv),
                                                         reduction_indices=[1]))) # same as loss 8, but with coin-specific consumptions.


        loss_function = loss_function5
        if self.__config["training"]["loss_function"] == "loss_function4":
            loss_function = loss_function4
        elif self.__config["training"]["loss_function"] == "loss_function5":
            loss_function = loss_function5
        elif self.__config["training"]["loss_function"] == "loss_function6":
            loss_function = loss_function6
        elif self.__config["training"]["loss_function"] == "loss_function7":
            loss_function = loss_function7
        elif self.__config["training"]["loss_function"] == "loss_function8":
            loss_function = with_last_w
        elif self.__config["training"]["loss_function"] == "loss_function9":
            loss_function = with_last_w

        loss_tensor = loss_function()
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_losses:
            for regularization_loss in regularization_losses:
                loss_tensor += regularization_loss
        return loss_tensor

    def init_train(self, learning_rate, decay_steps, decay_rate, training_method):
        learning_rate = tf.train.exponential_decay(learning_rate, self.__global_step,
                                                   decay_steps, decay_rate, staircase=False)
        #                                           decay_steps, decay_rate, staircase=True)
        if training_method == 'GradientDescent':
            train_step = tf.train.GradientDescentOptimizer(learning_rate).\
                         minimize(self.__loss, global_step=self.__global_step)
        elif training_method == 'Adam':
            train_step = tf.train.AdamOptimizer(learning_rate).\
                         minimize(self.__loss, global_step=self.__global_step)
        elif training_method == 'RMSProp':
            train_step = tf.train.RMSPropOptimizer(learning_rate).\
                         minimize(self.__loss, global_step=self.__global_step)
        else:
            raise ValueError()
        return train_step

    def train(self, x, y, last_w, setw):
        tflearn.is_training(True, self.__net.session)
        self.evaluate_tensors(x, y, last_w, setw, [self.__train_operation])

    def evaluate_tensors(self, x, y, last_w, setw, tensors):
        """
        :param x:
        :param y:
        :param last_w:
        :param setw: a function, pass the output w to it to fill the PVM
        :param tensors:
        :return:
        """
        tensors = list(tensors)
        tensors.append(self.__net.output)
        assert not np.any(np.isnan(x))
        assert not np.any(np.isnan(y))
        assert not np.any(np.isnan(last_w)),\
            "the last_w is {}".format(last_w)
#        logging.error('evaluate_tensors: len(x)=' + str(len(x)))
#        logging.error('evaluate_tensors: len(y)=' + str(len(y)))
#        logging.error('evaluate_tensors: len(last_w)=' + str(len(last_w)))
#        assert not len(x) == 2891
#        logging.error('evaluate_tensors: len(setw)=' + str(len(setw))) is a function
#        for tensor in tensors: is not tensors. It's operations.
#            logging.error('evaluate_tensors: tensor shape = ' + str(tensor.get_shape()))
#        logging.error('x shape is ' + str(x.shape))
#        logging.error('y shape is ' + str(y.shape))
#        logging.error('last_w shape is ' + str(last_w.shape))
#        logging.error('last_w shape is ' + str(last_w.shape) + ' and consumption_nparray shape is ' + str(self.consumption_nparray.shape))
        results = self.__net.session.run(tensors,
                                         feed_dict={self.__net.input_tensor: x,
                                                    self.__y: y,
                                                    self.__net.previous_w: last_w,
#                                                    self.__net.consumptions_vector: self.consumption_nparray,
                                                    self.__net.input_num: x.shape[0]})
        setw(results[-1][:, 1:])
        return results[:-1]

    # save the variables path including file name
    def save_model(self, path):
        logging.error("Saving model to " + path)
        self.__saver.save(self.__net.session, path)

    # The original single consumption value for all markets version
    def __pure_pc_c(self):
        c = 0.005 # self.__commission_ratio @poloniex (less closer to real costs, but less dead too.)
        # c = 0.01 # self.__commission_ratio @poloniex (closer to real costs)
        # c = 0.0025 # self.__commission_ratio @poloniex
        # c = 0.001  # self.__commission_ratio @binance
        # c = 0.000  # self.__commission_ratio @sanity
        w_t = self.__future_omega[:self.__net.input_num-1]  # rebalanced (-1 because the corrected previous period omega corresponds to the current pre-trade levels)
        w_t1 = self.__net.output[1:self.__net.input_num]    # But what's with the [1:...]?
        mu = 1 - tf.reduce_sum(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1)*c # Just a sec. Why are the omegas two dimensional?
        return mu

    # consumption vector (on each periods) - actually trading cost of the last transition.
    # Still not sure if network can learn that trading different assets has different costs.
    # Length of results is one less the number of assets, because BTC-BTC trade has no cost.
    def __pure_pc(self):
#        c = 0.0025 # self.__commission_ratio
#        return self.__pure_pc_c()
        cv = self.consumption_vector   # <-- best thing, but loss calc broken (WHY?)
        # need to use tf.tile to broadcast it to w's dims
#        ct = tf.tile(cv, tf.pack([1, self.__net.input_num - 1]))
#        logging.error('cv.size = ' + str(cv.size) + ', self.__net.input_num-1 shape = ' + str(self.__net.input_num-1))
#        ct = np.broadcast_to(cv, (self.consumption_vector.size, self.__net.input_num-1))
        w_t = self.__future_omega[:self.__net.input_num-1]  # rebalanced <--- use this
#        w_t = self.__future_omega[1:self.__net.input_num]  # rebalanced <--- testing, prolly wrong
#        w_t = self.__future_omega[:self.__net.input_num]  # rebalanced <--- testing, prolly wrong
        w_t1 = self.__net.output[1:self.__net.input_num]   # Orig. Use this.
#        w_t1 = self.__net.output[:self.__net.input_num]    # Experimental. Do ignore
#        w_t1 = self.__net.output[:self.__net.input_num-1]    # Experimental. Do ignore
#        ct = cv + tf.zeros(w_t.get_shape(), w_t.dtype) # too soon, shape isn't known yet.
#        ct = cv + tf.zeros(tf.shape(w_t[:, 1:]), w_t.dtype) # too soon, shape isn't known yet.
#        mu = 1 - tf.reduce_sum(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1)*c # Just a sec. Why are the omegas two dimensional
#        mu = 1 - tf.tensordot(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv, 1)   # Doesn't learn at all...
#        mu = 1 - tf.matmul(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv)   # Works, but generates [?, 1] instead of [?, ]
#        mu = 1 - tf.reduce_sum(tf.matmul(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv), axis=1)   # Why is this not a dot product?
        mu = 1 - tf.reduce_sum(tf.matmul(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv), axis=1)   # Orig. cv shape weirdosity is because matmul works only on rank 2 mats.
#        mu = 1 - tf.matmul(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv[:,0], axis=1)   # Why is this not a dot product?
        logging.error('w_t1 dims: ' + str(w_t1.get_shape())) # (?, 12)
        logging.error('w_t dims: ' + str(w_t.get_shape())) # (?, 12)
        logging.error('cv dims: ' + str(cv.get_shape())) # 
        logging.error('tf.reduce_sum(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1) shape: ' + str(tf.reduce_sum(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1).get_shape())) # (?, )
        logging.error('tf.tensordot(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv, 1) shape: ' + str(tf.tensordot(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv, 1).get_shape())) # <unknown>
        logging.error('tf.tensordot(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv, 0) shape: ' + str(tf.tensordot(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv, 1).get_shape())) # crash
        logging.error('tf.matmul(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv) shape: ' + str(tf.matmul(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv).get_shape())) #
        logging.error('tf.reduce_sum(tf.matmul(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv), axis=1) shape: ' + str(tf.reduce_sum(tf.matmul(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), cv), axis=1).get_shape())) #
        """
        mu = 1-3*c+c**2

        def recurse(mu0):
            factor1 = 1/(1 - c*w_t1[:, 0])
            if isinstance(mu0, float):
                mu0 = mu0
            else:
                mu0 = mu0[:, None]
            factor2 = 1 - c*w_t[:, 0] - (2*c - c**2)*tf.reduce_sum(
                tf.nn.relu(w_t[:, 1:] - mu0 * w_t1[:, 1:]), axis=1)
            return factor1*factor2

        for i in range(20):
            mu = recurse(mu)
        """
        return mu

    def set_consumption_vector (self, cv):
        logging.error('nnaget::set_consumption_vector -- ' + str(cv))
#        self.consumption_vector = tf.constant (np.transpose(np.broadcast_to(cv, (108, 11), np.float32)))
        self.consumption_vector = tf.constant (cv, dtype=np.float32, shape=(self.__coin_number,1), name='consumptions_vector') # The real one (but why (...,1)?!)
#        self.consumption_nparray = np.reshape(cv, (-1, 41))
#        self.consumption_nparray = cv * np.ones([109, 1])
#        self.consumption_vector = tf.constant(0.005, dtype=np.float32, shape=[self.__coin_number,1], name='consumptions_vector')  # TESTING! DO NOT USE!

    # the history is a 3d matrix, return a asset vector
    def decide_by_history(self, history, last_w):
        assert isinstance(history, np.ndarray),\
            "the history should be a numpy array, not %s" % type(history)
        assert not np.any(np.isnan(last_w))
        assert not np.any(np.isnan(history))
        logging.error("NNAget: history matrix: " + pprint.pformat(history)) 
        logging.error("NNAget: last omega: " + pprint.pformat(last_w)) 
        tflearn.is_training(False, self.session)
        history = history[np.newaxis, :, :, :]

        return np.squeeze(self.session.run(self.__net.output, feed_dict={self.__net.input_tensor: history,
                                                                         self.__net.previous_w: last_w[np.newaxis, 1:],
#                                                                         self.__net.consumptions_vector: self.consumption_nparray,
                                                                         self.__net.input_num: 1}))
