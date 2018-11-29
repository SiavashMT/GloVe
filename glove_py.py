from typing import List
from scipy import sparse
import numpy as np
from random import shuffle
import logging


def adagrad(weight, grad, sum_grad_squared, learning_rate: float):
    weight -= (learning_rate * grad / np.sqrt(sum_grad_squared))

    return weight


def sgd(weight, grad, learning_rate: float):
    weight -= learning_rate * grad

    return weight


def rmsprop(weight, grad, mean_square_weight, learning_rate: float, beta: float = 0.9):
    mean_square_weight = beta * mean_square_weight + (1. - beta) * grad ** 2
    weight -= (learning_rate * grad / np.sqrt(mean_square_weight))
    return weight, mean_square_weight


class GloVe:

    def __init__(self, window_size: int = 10, word_vector_dimension=100):

        self._window_size = window_size
        self._word_vector_dimension = word_vector_dimension

    def initialize_word_vectors_and_biases(self):

        self.W = (np.random.rand(self.vocab_size, self._word_vector_dimension) - 0.5) / \
                 float(self._word_vector_dimension + 1)
        self.W_tilde = (np.random.rand(self.vocab_size, self._word_vector_dimension) - 0.5) / \
                       float(self._word_vector_dimension + 1)

        self.b = (np.random.rand(self.vocab_size) - 0.5) / float(self._word_vector_dimension + 1)
        self.b_tilde = (np.random.rand(self.vocab_size) - 0.5) / float(self._word_vector_dimension + 1)

    @staticmethod
    def get_vocabs_from_corpus(corpus: List[List[str]]):
        return {word for doc in corpus for word in doc}

    def build_co_occurrence_matrix(self, corpus: List[List[str]], alpha: float = 0.75,
                                   x_max: int = 100):

        vocabs = GloVe.get_vocabs_from_corpus(corpus=corpus)

        id_to_word_map = {i: word for i, word in enumerate(vocabs)}
        word_to_id_map = {v: k for k, v in id_to_word_map.items()}

        vocab_size = len(vocabs)

        logging.info("Vocabulary size is {}".format(vocab_size))

        co_occurrence_matrix = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float64)

        for i, doc in enumerate(corpus):

            word_ids = [word_to_id_map[word] for word in doc]

            for center_i, center_id in enumerate(word_ids):

                context_ids = word_ids[max(0, center_i - self._window_size): center_i]
                contexts_len = len(context_ids)

                for left_i, left_id in enumerate(context_ids):
                    distance = contexts_len - left_i
                    increment = 1.0 / float(distance)

                    co_occurrence_matrix[center_id, left_id] += increment
                    co_occurrence_matrix[left_id, center_id] += increment

        self.co_occurrence_matrix = co_occurrence_matrix
        self.id_to_word_map = id_to_word_map
        self.word_to_id_map = word_to_id_map
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.x_max = x_max

        self.f_co_occurrence_matrix = self.f(x_max, alpha)
        self.log_co_occurrence_matrix = self.co_occurrence_matrix.tocsr().log1p()

    def f(self, x_max, alpha):

        def _f(x):
            return min(x / x_max, 1) ** alpha

        _f_vectorized = np.vectorize(_f, otypes=[np.float])

        result = self.co_occurrence_matrix.tocsr()
        result.data = _f_vectorized(self.co_occurrence_matrix.tocsr().data)
        return result

    def train(self, corpus: List[List[str]], number_of_iterations: int, alpha: float = 0.75,
              x_max: int = 100, optimizer: str = "sgd",
              learning_rate: float = 0.1, optimizer_params: dict = {}, display_iteration: int=1):

        self.build_co_occurrence_matrix(corpus=corpus, alpha=alpha, x_max=x_max)
        self.initialize_word_vectors_and_biases()

        id_pairs = [(x, y) for x, y in zip(*self.co_occurrence_matrix.nonzero())]

        if optimizer == 'adagrad':
            sum_grad_w_sq = np.zeros((self.vocab_size, self._word_vector_dimension))
            sum_grad_b_sq = np.zeros(self.vocab_size)
            sum_grad_w_tilde_sq = np.zeros((self.vocab_size, self._word_vector_dimension))
            sum_grad_b_tilde_sq = np.zeros(self.vocab_size)

        elif optimizer == 'rmsprop':
            mean_square_w = np.zeros((self.vocab_size, self._word_vector_dimension))
            mean_square_w_tilde = np.zeros((self.vocab_size, self._word_vector_dimension))
            mean_square_b = np.zeros(self.vocab_size)
            mean_square_b_tilde = np.zeros(self.vocab_size)

        for iteration in range(number_of_iterations):
            shuffle(id_pairs)

            total_cost = 0

            for i, j in id_pairs:
                inner_cost = (self.W[i, :].transpose().dot(self.W_tilde[j, :]) + self.b[i] + self.b_tilde[j] -
                              self.log_co_occurrence_matrix[i, j])

                grad_b_tilde_j = grad_b_i = self.f_co_occurrence_matrix[i, j] * inner_cost

                grad_w_i = self.W_tilde[j, :] * grad_b_i
                grad_w_tilde_j = self.W[i, :] * grad_b_tilde_j

                if optimizer == 'sgd':
                    self.b[i] = sgd(self.b[i], grad_b_i, learning_rate)
                    self.b_tilde[j] = sgd(self.b_tilde[j], grad_b_tilde_j, learning_rate)
                    self.W[i] = sgd(self.W[i], grad_w_i, learning_rate)
                    self.W_tilde[j] = sgd(self.W_tilde[j], grad_w_tilde_j, learning_rate)

                elif optimizer == 'adagrad':
                    sum_grad_w_sq[i] += grad_w_i ** 2
                    sum_grad_b_sq[i] += grad_b_i ** 2
                    sum_grad_w_tilde_sq[j] += grad_w_tilde_j ** 2
                    sum_grad_b_tilde_sq[j] += grad_b_tilde_j ** 2

                    self.b[i] = adagrad(self.b[i], grad_b_i, sum_grad_b_sq[i], learning_rate)
                    self.b_tilde[j] = adagrad(self.b_tilde[j], grad_b_tilde_j, sum_grad_b_tilde_sq[j], learning_rate)
                    self.W[i] = adagrad(self.W[i], grad_w_i, sum_grad_w_sq[i], learning_rate)
                    self.W_tilde[j] = adagrad(self.W_tilde[j], grad_w_tilde_j, sum_grad_w_tilde_sq[j], learning_rate)

                elif optimizer == 'rmsprop':
                    self.b[i], mean_square_b[i] = rmsprop(self.b[i], grad_b_i, mean_square_b[i], learning_rate,
                                                          **optimizer_params)
                    self.b_tilde[j], mean_square_b_tilde[j] = rmsprop(self.b_tilde[j], grad_b_tilde_j,
                                                                      mean_square_b_tilde[j], learning_rate,
                                                                      **optimizer_params)
                    self.W[i], mean_square_w[i] = rmsprop(self.W[i], grad_w_i, mean_square_w[i], learning_rate,
                                                          **optimizer_params)
                    self.W_tilde[j], mean_square_w_tilde[j] = rmsprop(self.W_tilde[j], grad_w_tilde_j,
                                                                      mean_square_w_tilde[j], learning_rate,
                                                                      **optimizer_params)

                cost = self.f_co_occurrence_matrix[i, j] * inner_cost

                total_cost += cost
            if iteration % display_iteration == 0:
                logging.info('Iteration: {}, Total Cost: {}'.format(iteration, total_cost))

    @property
    def word_mapping(self):
        return {self.id_to_word_map[i]: (self.W[i] + self.W_tilde[i])/2 for i in range(self.vocab_size)}
