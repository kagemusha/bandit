from visitor_model import VisitorModel
from incremental_cart.efdt import Efdt
from random import choice as rchoice
from data_generator import ClickGenerator
import random
import numpy as np
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


def online_ad_serving_test():
  print("Online ad serving test")
  batch_size = 6000
  batch_count = 15
  feature_names = ['sex', 'age']
  clickGen = get_click_gen3()
  num_choices = 2
  visit_params = [0.5, 0.5]
  models = []

  # run models
  for choice in range(num_choices):
    models.append(Efdt(feature_names, (0,1), delta=0.01, nmin=100, tau=0.5))
  batches = get_visitor_batches(batch_size, batch_count, feature_names, visit_params)

  types = [(0,0), (0,1), (1,0), (1,1)]
  # initial run with random probs
  print("========= Initial Run ===================")
  for choice in range(num_choices):
    batch = batches.pop(0)
    choices = [choice] * len(batch)
    clicks = [clickGen.get_click(visit, choice) for visit,choice in zip(batch, choices)]
    print_clicks(types, choice, num_choices, clicks, batch, choices)
    model = models[choice]
    print("Updating model...")
    for visit,click, in zip(batch, clicks):
      model.update(visit, click)
    for features in ((0,0), (0,1), (1,0), (1,1)):
      print("Model ", choice, ", ", features, " pred: ", "%.3f" % model.probs_single(features)[1])
    model.print_full_tree()
  print("========= End Initial Run ===================")
  correct_choices = [0,1,1,0] # automate this
  # run...
  for i, batch in enumerate(batches):
    choices = [get_choice_egreedy(.1, num_choices, models, visit) for visit in batch]
    clicks = [clickGen.get_click(visit, choice) for visit,choice in zip(batch, choices)]
    print_clicks(types, i, num_choices, clicks, batch, choices)
    for visit,choice,click, in zip(batch, choices, clicks):
      models[choice].update(visit, click)
    for i, features in enumerate([(0,0), (0,1), (1,0), (1,1)]):
      preds = []
      for k, model in enumerate(models):
        preds.append(model.probs_single(features)[1])
      pred_pct_str = " | ".join(["%.3f" % pred for pred in preds])
      served = np.argmax(preds)
      print(features, " pred: ", pred_pct_str, served, served == correct_choices[i])
    # model.print_full_tree()
  print("done")




def __main__():
  # we generate all batches up-front on assumption that what we serve has no affect on future visits
  # then we ret
  # test_dt3()
  online_ad_serving_test()

if __name__ == "__main__":
  __main__()
