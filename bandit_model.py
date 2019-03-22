from skmultiflow.trees import HATT
import random
from random import choice as rchoice
import numpy as np

class BanditModel:

  # todo: add model variations
  # todo: add explore-exploit strategies
  def __init__(self, num_choices, options):
    self.num_choices = num_choices
    self.choice_models = [HATT() for _ in range(num_choices)]
    self.options = options


  def get_choice(self, features):
    if random.random() < self.options['explore']:
      return rchoice(range(self.num_choices))
    else:
      feats = np.array(features).reshape(1,2)
      probs = [model.predict_proba(feats)[0] for model in self.choice_models]
      probs = [prob[1] if len(prob) == 2 else prob[0] for prob in probs]
      return np.argmax(probs)

  def train(self, choice, X, y):
    self.choice_models[choice].partial_fit(X, y)

  def predict_proba(self, features):
    features = np.array(features).reshape(1, len(features))
    return [m.predict_proba(features)[0] for m in self.choice_models]