import default_example as de
import numpy as np
from skmultiflow.trees import HATT
import time
from model_store import load_model_file, save_model_file

from util import print_clicks


def print_predictions(model):
  for features in ((0,0), (0,1), (1,0), (1,1)):
    feats = np.array(features).reshape(1,2)
    tim = time.time()
    pred = list(model.predict_proba(feats)[0])[1]
    endt = time.time()
    print("\t", features, " pred: ", '{:.1%}'.format(pred), "  time: ", endt-tim)

def basic_simulation():
  model = HATT()

  print("EFDT Basic Sim")
  num_choices = 2
  click_gen = de.default_sex_age_click_gen()
  de.test_default_sex_age_click_gen()

  batch_size = 1000
  batch_count = 10
  batches = de.get_batches(batch_size, batch_count)

  choice = 0
  for i, batch in enumerate(batches):
    choices = [choice] * len(batch)
    clicks = [click_gen.get_click(visit, choice) for visit,choice in zip(batch, choices)]
    print_clicks(de.types, i, clicks, batch, choices)
    tim = time.time()
    model.partial_fit(batch, clicks)
    print("fit time: ", time.time() - tim)
    print("--Model: ", choice)
    print_predictions(model)

def basic_sim_with_pickle():
  choice = 0
  save_model_file(choice, HATT())
  model = load_model_file(choice)

  print("EFDT Basic Pickled Model Sim")
  num_choices = 2
  clickGen = de.default_sex_age_click_gen()
  batch_size = 1000
  batch_count = 10
  batches = de.get_batches(batch_size, batch_count)

  for i, batch in enumerate(batches):
    choices = [choice] * len(batch)
    clicks = [clickGen.get_click(visit, choice) for visit,choice in zip(batch, choices)]
    print_clicks(de.types, i, clicks, batch, choices)
    tim = time.time()
    model.partial_fit(batch, clicks)
    save_model_file(choice, model)
    unpickled = load_model_file(choice)
    print("U==M?", unpickled == model)
    print("fit time: ", time.time() - tim)
    print("--Model: ", choice)
    print_predictions(unpickled)
    model = unpickled

basic_sim_with_pickle()
print("\n\n---------------- UNPICKLED PRED")
unpickled = load_model_file(0)
print_predictions(unpickled)

