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


class ClickGenerator:

  def __init__(self, base, features, propensity_rules, num_choices):
    self.base = base
    self.features = features
    self.propensity_rules = propensity_rules
    self.num_choices = num_choices
    self.propensity_pcts = self.__calc_propensity_pcts(base, propensity_rules, num_choices)


  def __calc_propensity_pcts(self, base, propensities, num_choices):
    propentsity_pcts = {}
    for i in range(2 ** len(self.features)):
      fmt = "{:0" + str(len(self.features)) + "b}"
      v_type = fmt.format(i)
      pcts = np.array([base] * num_choices)
      for rule, props in propensities.items():
        if self.rule_match(v_type, rule):
          pcts += props
      propentsity_pcts[v_type] = pcts
    return propentsity_pcts

    # update visitor types with rules, so should look like, e.g.
    #  (0,0,0): [0.04, 0.06]

  # v_type is a bin string such as '0100'
  def rule_match(self, v_type, rule):
    for i, prop in enumerate(rule):
      if prop:
        f = '0' if self.features[i]['vals'][0] == prop else '1'
        if f != v_type[i]:
          return False
    return True


  def get_propensities(self, obs):
    obs = ''.join(str(x) for x in obs)
    return self.propensity_pcts[obs]

def test():
  base = 0.04
  features = [
    { 'name': 'sex', 'vals': ["f","m"]},
    { 'name': 'placebo', 'vals': ["np","p"]},
    { 'name': 'age', 'vals': ["young","old"]},
  ]

  propensities = {
    ("f", None, None): [0.0, 0.01],
    ("f", None, "young"): [0.04, 0.0],
    ("m", None, None): [0.01, 0.0], # means males .01 more likely to click on choice 1 than general pop
    ("m", None, "old"): [0.04, 0.0],
    (None, None, "old"): [0.00, 0.02],
  }

  clickGen = ClickGenerator(base, features, propensities, 2)
  tests = [
    [[0,0,0], [0.08, 0.05]], # f young
    [[0,0,1], [0.04, 0.07]], # f old
    [[1,0,0], [0.05, 0.04]], # m young
    [[1,0,1], [0.09, 0.06]], # m old
  ]

  for atest in tests:
    props = clickGen.get_propensities(atest[0])
    print(props)
    assert(list(props) == atest[1])

if __name__ == "__main__":
  test()


def simulate_data(count):
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





