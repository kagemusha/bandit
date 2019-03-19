from visitor_model import VisitorModel
from random import choice as rchoice
from data_generator import ClickGenerator
import random
import numpy as np
from skmultiflow.trees import HATT
import time

from util import print_clicks

def get_click_gen3():
  base = 0.04
  features = [
    { 'name': 'sex', 'vals': ["f","m"]},
    { 'name': 'age', 'vals': ["young","old"]},
  ]

  propensities = {
    ("f", "young"): [0.04, 0.01],
    ("f", "old"): [0.0, 0.05],
    ("m", "young"): [0.01, 0.04], # means males .01 more likely to click on choice 1 than general pop
    ("m", "old"): [0.05, 0.0],
    # (None, "old"): [0.00, 0.02],
  }
  click_gen = ClickGenerator(base, features, propensities, 2)
  test_click_gen3(click_gen)

  return click_gen

def test_click_gen3(click_gen):
  tests = [
    [[0,0], [0.08, 0.05]], # f young
    [[0,1], [0.04, 0.09]], # f old
    [[1,0], [0.05, 0.08]], # m young
    [[1,1], [0.09, 0.04]], # m old
  ]

  print("Assertions...")
  for t in tests:
    props = click_gen.get_propensities(t[0])
    print(props)
    assert(list(props) == t[1])

def get_visitor_batches(batch_size, batch_count, param_names, visit_params):
  # means expect 30% of visitors sex[0] and 60% placebo[0]
  vModel = VisitorModel(param_names, visit_params)
  return [vModel.generate_visits(batch_size) for i in range(batch_count)]

def get_choice_egreedy(explore, num_choices, models, features):
  #e-greedy
  if random.random() < explore:
    return rchoice(range(num_choices))
  else:
    probs = [model.probs_single(features)[1] for model in models]
    return np.argmax(probs)
# 2. Instantiate the HoeffdingTree classifier

def basic_simulation():
  ht = HATT()

  print("MF Bandit")
  batch_size = 10000
  batch_count = 14
  feature_names = ['sex', 'age']
  clickGen = get_click_gen3()
  num_choices = 2
  visit_params = [0.5, 0.5]
  batches = get_visitor_batches(batch_size, batch_count, feature_names, visit_params)
  types = [(0,0), (0,1), (1,0), (1,1)]

  choice = 0
  for i, batch in enumerate(batches):
    choices = [choice] * len(batch)
    clicks = [clickGen.get_click(visit, choice) for visit,choice in zip(batch, choices)]
    print_clicks(types, i, num_choices, clicks, batch, choices)
    tim = time.time()
    ht.partial_fit(batch, clicks)
    print("fit time: ", time.time() - tim)
    print("--Model: ", choice)
    for features in ((0,0), (0,1), (1,0), (1,1)):
      feats = np.array(features).reshape(1,2)
      tim = time.time()
      pred = list(ht.predict_proba(feats)[0])[1]
      endt = time.time()
      print("\t", features, " pred: ", '{:.1%}'.format(pred), "  time: ", endt-tim)

basic_simulation()