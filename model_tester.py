import numpy as np
from model_store import load_model_file, save_model_file
import default_example as de

def print_preds(model):
  print("Printing preds")
  for features in de.types:
    feats = np.array(features).reshape(1,2)
    print("feats: ", feats, type(feats))
    pred = model.predict_proba(feats)
    print("raw pred: ", pred)
    pred = list(pred[0])[1]
    print("\t", features, " pred: ", '{:.1%}'.format(pred))
  # model.print_full_tree()
  print("-----\n")

print("+++ ModelTester +++")
for choice in range(2):
  print("Model: ", choice)
  model = load_model_file(choice)
  print_preds(model)