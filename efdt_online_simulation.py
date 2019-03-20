from random import choice as rchoice
import default_example as de
import random
import numpy as np
from util import print_clicks
from model_store import load_model_file, save_model_file
from skmultiflow.trees import HATT


def get_choice_egreedy(explore, num_choices, models, features):
  if random.random() < explore:
    return rchoice(range(num_choices))
  else:
    for model in models:
      print(type(model))
    feats = np.array(features).reshape(1,2)
    probs = [model.predict_proba(feats) for model in models]
    for prob in probs:
      print("GYG: ", prob)
    probs = [prob[0][1] for prob in probs]
    return np.argmax(probs)


def online_ad_serving_sim():
  print("EFDT Online ad serving sim!")
  batch_size = 3000
  batch_count = 5
  num_choices = 2

  # init models
  models = [HATT() for _ in range(num_choices)]
  # for choice in range(num_choices):
  #   model = HATT()
  #   save_model_file(choice, model)

  # get visit batches
  batches = de.get_batches(batch_size, batch_count)
  click_gen = de.default_sex_age_click_gen()

  # initial run with random probs
  print("\n========= Initial Run ===================")
  for choice in range(num_choices):
    batch = batches.pop(0)
    choices = [choice] * len(batch)
    clicks = [click_gen.get_click(visit, choice) for visit,choice in zip(batch, choices)]
    print_clicks(de.types, choice, clicks, batch, choices)
    model = models[choice]
    # model = load_model_file(choice)

    print(f"Updating model {choice}...")
    model.partial_fit(batch, clicks)
    for features in ((0,0), (0,1), (1,0), (1,1)):
      feats = np.array(features).reshape(1,2)
      pred = list(model.predict_proba(feats)[0])[1]
      print("\t", features, " pred: ", '{:.1%}'.format(pred))
    # model.print_full_tree()
  print("========= End Initial Run ===================\n")

  correct_choices = [0,1,1,0] # automate this

  # Batch runs
  for i, batch in enumerate(batches):
    # models = [ load_model_file(choice) for choice in range(num_choices)]
    choices = [get_choice_egreedy(.1, num_choices, models, visit) for visit in batch]
    clicks = [click_gen.get_click(visit, choice) for visit,choice in zip(batch, choices)]
    print_clicks(de.types, i, clicks, batch, choices)
    data = [ [[],[]] for _ in range(num_choices) ]
    for visit,choice,click, in zip(batch, choices, clicks):
      data[choice][0].append(visit)
      data[choice][1].append(click)
    for i in range(num_choices):
      models[i].partial_fit(data[i][0], data[i][1])
      # save_model_file(i, models[i])
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
  online_ad_serving_sim()

if __name__ == "__main__":
  __main__()
