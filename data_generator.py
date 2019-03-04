import numpy as np
import random
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import binom

class Results():


  def print(self):
    self.rate = 0.5
    results = f'''
      rate: {self.rate}
    '''
    print(results)

ctypes = ['a', 'b']
adtypes = ['a', 'b']

def click_prob(ctype, atype):
  return 0.05 if ctype == atype else 0.04


def predict(features):
  pass

def clicked(ctype, atype):
  return np.random.rand() < click_prob(ctype, atype)

# measuredPropensity = [
#   [0, 0.3, 1]
#   [0.3, 0.6, 2]
#   [0.6, 1.0, 3]
# ]

#features
# gender
# age: { < 25, 26 - 50, > 50 }
# propensity: 0,1,2
# placebo: .7 are +

# pop params
#

# CTRs
# m, <25: .08
# f, <25: .04
# m, 25-50: .05
# f, 25-50: .05
# m, >50: .03
# f, >50: .07

## case 1:
# males have .05 propensity for c1 .04 propensity for c2
# females have .04 propensity for c1 .05 propensity for c2


# propensity is at random and adds .002*num

def simulate_data(count):
  # flist = ['gender','placebo']
  malePct = .4
  placeboPct = .6
  maleClickPcts = [0.05, 0.04]
  femaleClickPcts = [0.04, 0.05]
  ds = []
  for i in range(count):
    sex = 1 if random.random() < malePct else 0
    p = 1 if random.random() < placeboPct else 0
    choice = random.randint(1,2)
    clickPcts = maleClickPcts if sex == 1 else femaleClickPcts
    propensity = clickPcts[choice-1]
    click = 1 if random.random() < propensity else 0
    ds.append([sex,p,choice,click])
  return ds

#



def profile_data(data):
  males = 0
  mclicks = np.array([[0,0],[0,0]])
  fclicks = np.array([[0,0],[0,0]])
  # placebo_clicks = [0,0]
  for obs in data:
    ary = mclicks if obs[0] == 1 else fclicks
    choice = obs[2] - 1
    click = obs[3]
    ary[choice][click] += 1

  for click_data in [mclicks, fclicks]:
    for c in click_data:
      tot = c[0] + c[1]
      print(f"{c[1]} clicks of {tot}: {c[1]/tot}%")

# batch_size = 1000
# data = simulate_data(batch_size)
# profile_data(data)

def generate_visits(count, visit_params):
  dataset = []
  for _ in range(count):
    obs = []
    for param in visit_params:
      val = 1 if random.random() < param else 0
      obs.append(val)
    dataset.append(obs)
  return dataset


count = 1000
visit_params = [0.3, 0.6]
ds = generate_visits(count, visit_params)

assert(len(ds) == count)
assert(len(ds[0]) == len(visit_params))
sums = np.sum(ds, axis=0)
print(sums.shape)
for i, col in enumerate(ds[0]):
  assert(col ==1 or col == 0)
  pct = visit_params[i]
  (lo, hi) = binom.interval(.954, 1000, pct)
  print("sums", i, lo, sums[i], hi)
  assert(lo <= sums[i] <= hi)





