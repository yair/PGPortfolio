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
        logging.debug("a new experience, indexed by %d, was appended" % state_index)

    def __sample(self, start, end, bias):
        return self.__sample_aug_linear(start, end, bias)

    def __sample_unaug(self, start, end, bias):
        """
        @:param end: is excluded
        @:param bias: value in (0, 1)
        """
        # TODO: deal with the case when bias is 0 (what does that even mean?)
        ran = np.random.geometric(bias)
        while ran > end - start:
            ran = np.random.geometric(bias)     # TODO: Don't use geometric (or adapt it properly) if using oversampling.
        result = end - ran
        return result

    def __sample_aug_linear(self, start, end, bias): # TODO: This is not linear

        if (self.__aug_factor == 1):
            return self.__sample_unaug(start, end, bias) # Yeah, it's not linear then, whatever.

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

        while True:
#            ran = np.random(start, end)
            ran = np.random.geometric(bias / self.__aug_factor)
            if ran > end - start: # out of bounds (both testing and training)
                continue
            # Excluding both training tail end and... no, actually, we can use testing batches from the start.
            if ran >= self.initial_end - self.__aug_factor * self.__batch_size and ran < self.initial_end: # + self.__batch_size:
                continue
            if ran > self.initial_end:
#                logging.error('Training on testing batch no. ' + str(ran-self.initial_end) + '. Contents: ' + str(map(lambda x:x.state_index, self.__experiences[ran:ran + self.__batch_size * self.__aug_factor:self.__aug_factor])))
                logging.error('Training on testing batch no. ' + str(ran-self.initial_end) + '. Contents: ')
                map(lambda x:logging.error(str(x.state_index)), self.__experiences[ran:ran + self.__batch_size * self.__aug_factor:self.__aug_factor])
            return ran

    def next_experience_batch(self):
        # First get a start point randomly
        batch = []
#        if self.__aug_factor > 1:
#            batch_start = self.__sample(0, len(self.__experiences) - self.__batch_size, self.__sample_bias)
#            batch = self.__experiences
        if self.__is_permed:
#            assert False, 'Please do not check is_permed, it kills the previous w feature'
            for i in range(self.__batch_size):
                batch.append(self.__experiences[self.__sample(self.__experiences[0].state_index,
                                                              self.__experiences[-1].state_index,
                                                              self.__sample_bias)])
        else:
            batch_start = self.__sample(0, len(self.__experiences) - self.__batch_size,     # TODO: This is what needs to be hacked to make oversampling work.
                                        self.__sample_bias)
            batch = self.__experiences[batch_start:batch_start + self.__batch_size * self.__aug_factor:self.__aug_factor]
#            batch_start = self.__sample(0, len(self.__experiences) - self.__batch_size,     # TODO: This is what needs to be hacked to make oversampling work.
#                                        self.__sample_bias)
#            batch = self.__experiences[batch_start:batch_start + self.__batch_size]
        return batch


class Experience:
    def __init__(self, state_index):
        self.state_index = int(state_index)
