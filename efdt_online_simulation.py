import default_example as de
import numpy as np
from util import print_clicks
from model_store import load_model, pickle_model
from bandit_model import BanditModel

MODEL_FILE = "_efdt_bandit_model"
checkmark = u'\u2713'

def online_ad_serving_sim():
  print("EFDT Online ad serving sim!")
  batch_size = 3000
  batch_count = 10
  num_choices = 2

  # init models
  model = BanditModel(2, {'explore': .1})
  pickle_model(MODEL_FILE, model)


  # get visit batches
  batches = de.get_batches(batch_size, batch_count)
  click_gen = de.default_sex_age_click_gen()

  # initial run with random probs
  print("\n========= Initial Run ===================\n")
  for choice in range(num_choices):
    batch = batches.pop(0)
    choices = [choice] * len(batch)
    clicks = [click_gen.get_click(visit, choice) for visit,choice in zip(batch, choices)]
    print_clicks(de.types, choice, clicks, batch, choices)
    model = load_model(MODEL_FILE)

    print(f"\nUpdating model {choice}...")
    model.train(choice, batch, clicks)
    print_preds(model, de.expected_choices)
  pickle_model(MODEL_FILE, model)
  print("\n========= End Initial Run ===================\n")


  # Batch runs
  for i, batch in enumerate(batches):
    model = load_model(MODEL_FILE)
    choices = [model.get_choice(visit) for visit in batch]
    clicks = [click_gen.get_click(visit, choice) for visit,choice in zip(batch, choices)]
    print_clicks(de.types, i, clicks, batch, choices)
    data = [ [[],[]] for _ in range(num_choices) ]
    for visit,choice,click, in zip(batch, choices, clicks):
      data[choice][0].append(visit)
      data[choice][1].append(click)
    for ch in range(num_choices):
      model.train(ch, data[ch][0], data[ch][1])
    pickle_model(MODEL_FILE, model)
    print_preds(model, de.expected_choices)
    # model.print_full_tree()
  print("done")

def print_preds(model, expected_choices):
  print("Printing preds")
  for features, expected in zip(de.types, expected_choices):
    preds = list(model.predict_proba(features))
    preds = [pred[1] if len(pred)>1 else pred[0] for pred in preds]
    print("Features: ", features)


    for i, pred in enumerate(preds):

      print("\tModel ", {i}, ": ", '{:.1%}'.format(pred))
    served = np.argmax(preds)
    print(f"Served: {served} Expected: {expected} {checkmark if served == expected else 'x'}")
  # model.print_full_tree()
  print("-----\n")


def __main__():
  # we generate all batches up-front on assumption that what we serve has no affect on future visits
  online_ad_serving_sim()

if __name__ == "__main__":
  __main__()
