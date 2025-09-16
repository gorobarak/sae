import torch
import torch.nn.functional as F
import numpy as np

class ShuffledDPSanitizer:

  def __init__(self, delta=10e-6):
    self.delta0 = 0.5 * delta
    self.delta_prime_composition = 0.5 * delta
    self.total_delta = self.delta0 + self.delta_prime_composition

  def sanitize_counter_array(self, counter_array, n, sensitivity, epsilon):
    """
    counter_array: array of non-negative integers.
    n:  number of records (users) represented in the counter array.
        Note: This parameter is currently not used in the code.
    sensitivity: how many entries each user at most contributed to.
        If set to 0, we assume that each user contributed to all entries.
    epsilon: input (not output!) privacy parameter.
    """
    num_counters = counter_array.size(0)
    # Set privacy parameters without sparsity
    if sensitivity == 0:
      epsilon_per_counter = epsilon / num_counters
      delta_per_counter = self.delta0 / num_counters
      output_epsilon = epsilon_per_counter * (np.e ** epsilon_per_counter - 1) * num_counters + \
                          epsilon_per_counter * (2 * num_counters * np.log(1. / self.delta_prime_composition)) ** 0.5
    else:
       epsilon_per_counter = epsilon / sensitivity
       delta_per_counter = self.delta0 / sensitivity
       output_epsilon = epsilon

    # Impose privacy noise
    negbin_param_r = 1.
    try:
      negbin_param_p = 1 - np.e ** (-0.8 * epsilon_per_counter)
    except:
      negbin_param_p = 0
    # Using numpy for negative binomial distribution since it isn't clear how pytorch parameterized it...
    negbin_sample1 = torch.from_numpy(np.random.negative_binomial(negbin_param_r, negbin_param_p, num_counters))
    negbin_sample2 = torch.from_numpy(np.random.negative_binomial(negbin_param_r, negbin_param_p, num_counters))
    noisy_counter_array = counter_array + negbin_sample1 - negbin_sample2

    return noisy_counter_array,output_epsilon



class ShuffledDPHistogramSanitizer:

  def __init__(self):
    self.shuffled_dp_sanitizer = ShuffledDPSanitizer()

  def sanitize(self, histogram, n, k, epsilon, use_sparsity=True):
    """
    histogram: input histogram.
    n:  number of records (users) represented in the counter array.
    k: how many histogram entries each user at most contributed to.
    epsilon: input (not output!) privacy parameter.
    """
    if use_sparsity:
      return self.shuffled_dp_sanitizer.sanitize_counter_array(histogram, n, k, epsilon)
    else:
      return self.shuffled_dp_sanitizer.sanitize_counter_array(histogram, n, 0, epsilon)


class ShuffledDPMeanSanitizer:

  def __init__(self, dataset):
    """
    dataset: 2D pytorch tensor with each row assumed to be already normalized to unit length.
    """
    self.n = dataset.size(0)
    self.d = dataset.size(1)

    rr_mask = torch.rand(dataset.size())
    rr_dataset = 0.5 * dataset + 0.5
    rr_dataset[rr_mask > rr_dataset] = 0
    rr_dataset[rr_mask <= rr_dataset] = 1
    self.counter_array = torch.sum(rr_dataset, dim=0)

    self.shuffled_dp_sanitizer = ShuffledDPSanitizer()

  def sanitize(self, epsilon):
    """
    epsilon: input (not output!) privacy parameter.
    """
    noisy_counter_array, output_epsilon = self.shuffled_dp_sanitizer.sanitize_counter_array(self.counter_array,
                                                                                            self.n,
                                                                                            0,
                                                                                            epsilon)
    noisy_mean_array = 2 * noisy_counter_array / self.n - 1
    return noisy_mean_array, output_epsilon
