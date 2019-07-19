from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import logging


class ReplayBuffer:
    def __init__(self, start_index, end_index, batch_size, is_permed, coin_number, sample_bias=1.0, aug_factor=1):
        """
        :param start_index: start index of the training set on the global data matrices
        :param end_index: end index of the training set on the global data matrices
        """
        self.__coin_number = coin_number
        self.__aug_factor = aug_factor
        logging.error('ReplayBuffer(start_index=' + str(start_index) + ' end_index=' + str(end_index) + ' batch_size=' + str(batch_size) + ' is_permed=' + str(is_permed) + ' sample_bias=' + str(sample_bias) + ' aug_factor=' + str(aug_factor))
#        self.__experiences = [Experience(i) for i in range(start_index, end_index)]
        self.initial_size = end_index - start_index # Already augged. was: start_index + aug_factor * (end_index - start_index)
        self.initial_end = end_index
        self.initial_start = start_index
        self.initial_unauged_size = (end_index - start_index) // aug_factor
        logging.error('ReplayBuffer initialized with ' + str(self.initial_size) + ' experiences')
        self.__experiences = [Experience(i) for i in range(start_index, start_index + self.initial_size)]
        self.__is_permed = is_permed
        # NOTE: in order to achieve the previous w feature
        self.__batch_size = batch_size
        self.__sample_bias = sample_bias
        logging.error("buffer_bias is %f" % sample_bias)

    def append_experience(self, state_index):   # Called from rolling trainer.
        self.__experiences.append(Experience(state_index))
        logging.error("a new experience, indexed by %d, was appended" % state_index)

    def __sample(self, start, end, bias):
#        if self.__aug_factor == 1:
#            return self.__sample_unaug(start, end, bias) # Yeah, it's not linear then, whatever.
#        else:
            return self.__sample_aug(start, end, bias)

    def __sample_unaug(self, start, end, bias):
        """
        @:param end: is excluded
        @:param bias: value in (0, 1)
        """
        # TODO: deal with the case when bias is 0 (what does that even mean?)

        assert False, 'Needs to be modified to filter insead of next_experience_batch'

        ran = np.random.geometric(bias)
        while ran > end - start:
            ran = np.random.geometric(bias)     # TODO: Don't use geometric (or adapt it properly) if using oversampling.
        result = end - ran
        logging.error('__sample_unaug: start=' + str(start) + ' bias=' + str(bias) + ' result=(end-ran)=' + str(end) + '-' + str(ran) + '=' + str(end-ran))
        return result

    def __sample_aug(self, start, end, bias): # TODO: unify both methods. The augged one should work with factor 1

        debug = False
        # Interleaved scheme, but this is not how our indices are organized.
#        +++++++++++++++++++++-------
#        +++++++++++++++++++++------- augged train data
#        +++++++++++++++++++++-------
#        +++++++++++++++++++++--------------+++++++  test data
#         Unaugged train data

        # What we have
        #++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++ un- and augged train data - usable
        #+++++++++++++++++++++++++---
        #-----------------------------------+++++++-------
        # Unusable train data           Un- and usable test data

        # Eventually we'll want to interleave and sextuple testing as well, then all will be usable (except absolute tails).

        # Firstly, the whole geometric sampling thing is broken here because time moves aug_factor times faster (per experience) during testing.
        # To solve this, we'll pretend we have aug_factor times the number of testing experiences, and then map it back if we sample one of these.

        testing_experiences = len(self.__experiences) - self.initial_end
        #pseudo_exlen = self.initial_end + self.__aug_factor * (len(self.__experiences) - self.initial_end)
        pseudo_exlen = self.initial_end + self.__aug_factor * (testing_experiences)
        aug_batch_size = self.__batch_size * self.__aug_factor
        # exclusion_zone_size = aug_batch_size + window_size? Why not? No leaky leaky otherwise? How did work before? Is that the dm backoff?
        if debug:
            logging.error('__sample_aug: testing_experiences=' + str(testing_experiences) + ' pseudo_exlen=' + str(pseudo_exlen) + ' initial_end=' + str(self.initial_end) + ' aug_batch_size=' + str(aug_batch_size))

        while True:

            psample = pseudo_exlen - np.random.geometric(bias / self.__aug_factor) # [-inf, pseudo_exlen]

            # Exclusion zones are areas where a batch constructed from any of their members as beginning will violate causality
            # | test_excl | test | train_excl | train | before the beginning of time |

#174 psample < 0 => try again                                    # this is before our experience begins
#175 psample < initial_end - aug_batch_size => return psample    # This is simply a training sample with full aug access - must end before training range ends
#176 psample < initial_end  => try again                         # Batches that cross from training to testing ranges contaminate the future
#177 psample < pseudo_exlen - aug_batch_size => tsamp = (psample-initial_end)/aug_factor; return (initial_end + tsamp)
#178 psample < pseudo_exlen => try again                         # We don't have enough samples before we cross into the future.
#179 else Assert false

            if psample < 0:     # TODO: If initial_end != len(self.__experiences), we're not in training and can use all of [0, pseudo_exlen - aug_batch_size]
                if debug:
                    logging.debug('__sample_aug: Skipping psample no. ' + str(psample) + ' - it is before the beginning of time.')
                continue
            if psample < self.initial_end - aug_batch_size:
                if debug:
                    logging.error('__sample_aug: Accpeting psample no. ' + str(psample) + ' - it is a training experience.')
                return psample
            if psample < self.initial_end:
                if debug:
                    logging.error('__sample_aug: Skipping psample no. ' + str(psample) + ' - it is in the training exclusion zone.')
                continue
            if psample < pseudo_exlen - aug_batch_size:
                test_period = (psample - self.initial_end) // self.__aug_factor
                logging.error('__sample_aug: Accpeting testing psample no. ' + str(psample) + ' (from test period ' + str(test_period) + ')')
                return self.initial_end + test_period
            if psample < pseudo_exlen:
                if debug:
                    logging.error('__sample_aug: Skipping psample no. ' + str(psample) + ' - it is in the testing exclusion zone.')
                continue
            assert False, '__sample_aug: Invalid pseudo sample no. ' + str(psample)


        """ This is all wrong. We should be returning end-ran, the thick end is near the present, not the beginning of time. Can explain our preference for low biases.
        while True:
#            ran = np.random(start, end)
            ran = np.random.geometric(bias / self.__aug_factor)
            if ran > end - start: # out of bounds (both testing and training)
                logging.error('__sample_aug: ran>end-start (' + str(ran) + '>' + str(end) + '-' + str(end) + '). Trying again')
                continue
            # Excluding both training tail end and... no, actually, we can use testing batches from the start.
            if ran >= self.initial_end - self.__aug_factor * self.__batch_size and ran < self.initial_end: # + self.__batch_size:
                logging.error('__sample_aug: ran>end-start (' + str(ran) + '>' + str(end) + '-' + str(end) + '). Trying again')
                continue
            if ran > self.initial_end:
#                logging.error('Training on testing batch no. ' + str(ran-self.initial_end) + '. Contents: ' + str(map(lambda x:x.state_index, self.__experiences[ran:ran + self.__batch_size * self.__aug_factor:self.__aug_factor])))
                logging.error('Training on testing batch no. ' + str(ran-self.initial_end) + '. Contents: ')
                map(lambda x:logging.error(str(x.state_index)), self.__experiences[ran:ran + self.__batch_size * self.__aug_factor:self.__aug_factor])
            return ran
        """

    def next_experience_batch(self):
        # First get a start point randomly
        batch = []
#        if self.__aug_factor > 1:
#            batch_start = self.__sample(0, len(self.__experiences) - self.__batch_size, self.__sample_bias)
#            batch = self.__experiences
        if self.__is_permed:
            assert False, 'Please do not check is_permed, it kills the previous w feature'
            for i in range(self.__batch_size):
                batch.append(self.__experiences[self.__sample(self.__experiences[0].state_index,
                                                              self.__experiences[-1].state_index,
                                                              self.__sample_bias)])
        else:
            # This is a mess. We are filtering twice. Let sample do all the filtering. We only need to make sure training batches are augged and testing not.
            batch_start = self.__sample(0, len(self.__experiences), self.__sample_bias)
            if (batch_start > self.initial_end):    # testing
                batch = self.__experiences[batch_start : batch_start + self.__batch_size] # Boog, BTW, training and testing should have diff. batch sizes
            else:                                   # training
                batch = self.__experiences[batch_start : batch_start + self.__batch_size * self.__aug_factor : self.__aug_factor]
#            batch_start = self.__sample(0, len(self.__experiences) - self.__batch_size,     # TODO: This is what needs to be hacked to make oversampling work.
#                                        self.__sample_bias)
#            batch = self.__experiences[batch_start:batch_start + self.__batch_size * self.__aug_factor:self.__aug_factor]
#            batch_start = self.__sample(0, len(self.__experiences) - self.__batch_size,     # TODO: This is what needs to be hacked to make oversampling work.
#                                        self.__sample_bias)
#            batch = self.__experiences[batch_start:batch_start + self.__batch_size]
        return batch


class Experience:
    def __init__(self, state_index):
        self.state_index = int(state_index)
