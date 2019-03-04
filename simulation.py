import random
from visitor_model import VisitorModel
from random_model import RandomModel
from efdt import Efdt

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
  maleClickPcts = [0.2, 0.2]
  femaleClickPcts = [0.1, 0.1]
  clicks = []
  for i,row in enumerate(batch):
    sex = row[0]
    choice = choices[i]
    clickPcts = maleClickPcts if sex == 1 else femaleClickPcts
    propensity = clickPcts[choice-1]
    click = 1 if random.random() < propensity else 0
    clicks.append(click)
  return clicks

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

def get_visitor_batches(batch_size, batch_count):
  # means expect 30% of visitors sex[0] and 60% placebo[0]
  param_names = ['sex','placebo']
  visit_params = [0.3, 0.6]

  vModel = VisitorModel(param_names, visit_params)
  return [vModel.generate_visits(batch_size) for i in range(batch_count)]


def simulate_dt():
  batch_size = 10000
  batch_count = 5
  batches = get_visitor_batches(batch_size, batch_count)
  ctrs = []
  features = ['sex','placebo']
  model = Efdt(features, delta=0.01, nmin=100, tau=0.5)
  for i, batch in enumerate(batches):
    choices = model.get_batch_predictions(batch)
    results = get_clicks(choices, batch)
    ctr = get_ctr(results)
    print(f"Batch {i}: ct: {ctr}")
    ctrs.append(ctr)
    model.retrain(results)
  ctr = 100 * (sum(ctrs)/len(ctrs))
  print("overall ctr: %.2f" % ctr)


def __main__():
  # we generate all batches up-front on assumption that what we serve has no affect on future visits
  # then we ret
  simulate_random()

if __name__ == "__main__":
  __main__()
