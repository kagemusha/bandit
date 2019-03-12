#
#  how does the bandit work
#  - the y states are click and no-click
#  - each choice has a tree
#  - for prediction, you run the choice thru each tree and take the
#  - choice which offers the highest
#
#
#
#

from itertools import product
from visitor_model import VisitorModel
from random_model import RandomModel
from incremental_cart.efdt import Efdt
from random import choice as rchoice
from data_generator import ClickGenerator
import random
import numpy as np

class Result:
  # order_num
  # click-thru by group
  # prediction by group
  # served by group
  # pre-retrain-probs
  # post-retrain-probs
  def __init__(self):
    pass

class Batch:
  def __init__(self, input_model, order):
    self.order = order
    input_model.generate_batch()


def get_ctr(results):
  lenr = len(results)
  clicks = sum(results)
  return clicks/lenr


def get_clicks(choices, batch):
  maleClickPcts = [0.10, 0.05]
  femaleClickPcts = [0.05, 0.1]
  clicks = []
  # print(f"Click propensities for F(sex=0): {femaleClickPcts}")
  # print(f"Click propensities for M(sex=1): {maleClickPcts}")
  for i,row in enumerate(batch):
    sex = row[0]
    choice = choices[i]
    clickPcts = maleClickPcts if sex == 1 else femaleClickPcts
    propensity = clickPcts[choice-1]
    click = 1 if random.random() < propensity else 0
    clicks.append(click)
  return clicks




def get_visitor_batches(batch_size, batch_count, param_names, visit_params):
  # means expect 30% of visitors sex[0] and 60% placebo[0]

  vModel = VisitorModel(param_names, visit_params)
  return [vModel.generate_visits(batch_size) for i in range(batch_count)]


def get_batch_predictions(model, batch, choices):

  preds = []
  # eventually add things like thompson sampling. for now epsilon greedy with e=.1
  for obs in batch:
    if random.random() < 0.1:
      preds.append(rchoice(choices))
    else:
      model.get
  # get click pct for each choice model
  # choose max

def simulate_dt():
  batch_size = 1000
  batch_count = 2
  batches = get_visitor_batches(batch_size, batch_count)
  ctrs = []
  features = ['sex','placebo']
  model = Efdt(features, (0,1), delta=0.001, nmin=100, tau=0.5)
  for i, batch in enumerate(batches):
    choices = get_batch_predictions(model, batch)
    results = get_clicks(choices, batch)
    ctr = get_ctr(results)
    print(f"Batch {i}: ct: {ctr}")
    ctrs.append(ctr)
    model.retrain(results)
  ctr = 100 * (sum(ctrs)/len(ctrs))
  print("overall ctr: %.2f" % ctr)

def simulate_random():
  batch_size = 10000
  batch_count = 5
  batches = get_visitor_batches(batch_size, batch_count)
  print("Random model")
  choice_count = 2
  model = RandomModel(choice_count)
  ctrs = []
  for i, batch in enumerate(batches):
    choices = model.get_batch_predictions(batch)
    results = get_clicks(choices, batch)
    ctr = get_ctr(results)
    print(f"Batch {i}: ct: {ctr}")
    ctrs.append(ctr)
    model.retrain(results)
  ctr = 100 * (sum(ctrs)/len(ctrs))
  print("overall ctr: %.2f" % ctr)

def clicks_by_sex(clicks, batch):
  results = {}
  for sex in (0,1):
    click_count = 0
    s_count = 0
    for c,obs in zip(clicks, batch):
      if obs[0] == sex:
        s_count += 1
        if c == 1:
          click_count += 1
    results[sex] = {'clicks': click_count, 'tot': s_count, 'pct': click_count/s_count}
  return results


def test_dt():
  # generate visitors
  print("test Dec Tree")
  batch_size = 1000
  batch_count = 10
  feature_names = ['sex','placebo']
  visit_params = [0.3, 0.6]
  batches = get_visitor_batches(batch_size, batch_count, feature_names, visit_params)
  batch = batches[0]
  # get results
  models = [Efdt(feature_names, (0,1), delta=0.01, nmin=100, tau=0.5) for i in range(2)]
  print("Feature names: ", feature_names)
  for ib, batch in enumerate(batches):

    print(f"========================Batch {ib}==================================")
    for i in (1,2):
      model = models[i-1]
      print("---------")
      print("Choice ", i)
      choice_served = [i] * len(batch)
      clicks = get_clicks(choice_served, batch)
      print(clicks_by_sex(clicks, batch))
      for x,y in zip(batch, clicks):
        model.update(x, y)

      model.print_full_tree()
      for features in list(product((0,1), (0,1))):
        print("pred: ", features, model.probs_single(features))

def get_click_gen():
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

  return ClickGenerator(base, features, propensities, 2)

def test_dt2():
  # generate visitors
  print("test Dec Tree 2")
  batch_size = 1000
  batch_count = 30
  feature_names = ['sex', 'placebo', 'age']
  clickGen = get_click_gen()
  visit_params = [0.5, 0.5, 0.5]
  batches = get_visitor_batches(batch_size, batch_count, feature_names, visit_params)

  # get results
  for i in (1,2):
    print("---------")
    print("Choice ", i)
    print("Feature names: ", feature_names)
    model = Efdt(feature_names, (0,1), delta=0.01, nmin=100, tau=0.5)

    for bi, batch in enumerate(batches):
      print(f"========Choice {i} Batch {bi}=======================")
      choice_served = i
      clicks = [clickGen.get_click(obs, choice_served) for obs in batch]
      for x,y in zip(batch, clicks):
        model.update(x, y)

      model.print_full_tree()
      for features in ((0,0,0), (0,0,1), (1,0,0), (1,0,1)):
        print(features, " pred: ", "%.3f" % model.probs_single(features)[1])

def get_click_gen3():
  base = 0.04
  features = [
    { 'name': 'sex', 'vals': ["f","m"]},
    { 'name': 'age', 'vals': ["young","old"]},
  ]

  propensities = {
    ("f", None): [0.0, 0.01],
    ("f", "young"): [0.04, 0.0],
    ("m", None): [0.01, 0.0], # means males .01 more likely to click on choice 1 than general pop
    ("m", "old"): [0.04, 0.0],
    (None, "old"): [0.00, 0.02],
  }

  return ClickGenerator(base, features, propensities, 2)

def test_dt3():
  # generate visitors
  print("test dec tree 2")
  batch_size = 1000
  batch_count = 30
  feature_names = ['sex', 'age']
  click_gen = get_click_gen3()
  visit_params = [0.5, 0.5]
  batches = get_visitor_batches(batch_size, batch_count, feature_names, visit_params)

  # get results
  for i in (1,2):
    print("---------")
    print("Choice ", i)
    print("Feature names: ", feature_names)
    model = Efdt(feature_names, (0,1), delta=0.01, nmin=100, tau=0.5)

    for bi, batch in enumerate(batches):
      print(f"========Choice {i} Batch {bi}=======================")
      choice_served = i
      clicks = [click_gen.get_click(obs, choice_served) for obs in batch]
      for x,y in zip(batch, clicks):
        model.update(x, y)

      model.print_full_tree()
      for features in ((0,0), (0,1), (1,0), (1,1)):
        print(features, " pred: ", "%.3f" % model.probs_single(features)[1])
#  want
#   1       2
#  .08     .05
#  .04     .07
#  .05     .04
#  .09     .06

  # feed into 2 decision trees
  # get predictions from each




def __main__():
  # we generate all batches up-front on assumption that what we serve has no affect on future visits
  # then we ret
  test_dt()

if __name__ == "__main__":
  __main__()
