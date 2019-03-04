import random
import numpy as np
from scipy.stats import binom

class VisitorModel:
  def __init__(self, names, visitor_params):
    self.names = names
    self.visitor_params = visitor_params


  def generate_visits(self, count):
    dataset = []
    for _ in range(count):
      obs = []
      for param in self.visitor_params:
        val = 1 if random.random() < param else 0
        obs.append(val)
      dataset.append(obs)
    return dataset

if __name__ == "__main__":
  count = 1000
  visitor_params = [0.3, 0.6]
  vm = VisitorModel(["a","b"], visitor_params)

  ds = vm.generate_visits(count)

  assert(len(ds) == count)
  assert(len(ds[0]) == len(visitor_params))
  sums = np.sum(ds, axis=0)
  print(sums.shape)
  for i, col in enumerate(ds[0]):
    assert(col ==1 or col == 0)
    pct = visitor_params[i]
    (lo, hi) = binom.interval(.954, 1000, pct)
    print("sums", i, lo, sums[i], hi)
    assert(lo <= sums[i] <= hi)

