from visitor_model import VisitorModel
from random import choice as rchoice
from data_generator import ClickGenerator
import random
import numpy as np
from util import print_clicks

from skmultiflow.trees import HATT

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
    feats = np.array(features).reshape(1,2)
    probs = [model.predict_proba(feats)[0][1] for model in models]
    return np.argmax(probs)


def online_ad_serving_test():
  print("Online ad serving test")
  batch_size = 5000
  batch_count = 15
  feature_names = ['sex', 'age']
  clickGen = get_click_gen3()
  num_choices = 2
  visit_params = [0.5, 0.5]
  models = []
  types = [(0,0), (0,1), (1,0), (1,1)]

  # init models
  for choice in range(num_choices):
    models.append(HATT())

  # get visit batches
  batches = get_visitor_batches(batch_size, batch_count, feature_names, visit_params)

  # initial run with random probs
  print("========= Initial Run ===================")
  for choice in range(num_choices):
    batch = batches.pop(0)
    choices = [choice] * len(batch)
    clicks = [clickGen.get_click(visit, choice) for visit,choice in zip(batch, choices)]
    print_clicks(types, choice, num_choices, clicks, batch, choices)
    model = models[choice]
    print(f"Updating model {choice}...")
    model.partial_fit(batch, clicks)
    for features in ((0,0), (0,1), (1,0), (1,1)):
      feats = np.array(features).reshape(1,2)
      pred = list(model.predict_proba(feats)[0])[1]
      print("\t", features, " pred: ", '{:.1%}'.format(pred))
    # model.print_full_tree()
  print("========= End Initial Run ===================")

  correct_choices = [0,1,1,0] # automate this
  # run...
  for i, batch in enumerate(batches):
    choices = [get_choice_egreedy(.1, num_choices, models, visit) for visit in batch]
    clicks = [clickGen.get_click(visit, choice) for visit,choice in zip(batch, choices)]
    print_clicks(types, i, num_choices, clicks, batch, choices)
    data = [ [[],[]] for _ in range(num_choices) ]
    for visit,choice,click, in zip(batch, choices, clicks):
      data[choice][0].append(visit)
      data[choice][1].append(click)
    for i in range(num_choices):
      models[i].partial_fit(data[i][0], data[i][1])
    for i, features in enumerate([(0,0), (0,1), (1,0), (1,1)]):
      preds = []
      for k, model in enumerate(models):
        feats = np.array(features).reshape(1,2)
        preds.append(model.predict_proba(feats)[0][1])
      pred_pct_str = " | ".join(['{:.1%}'.format(pred) for pred in preds])
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
