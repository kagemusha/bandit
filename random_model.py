import random

class RandomModel():
  def __init__(self, choice_count):
    self.choice_count = choice_count

  def retrain(self, results):
    return self

  def get_batch_predictions(self, batch):
    return [self.get_choice(row) for row in batch]

  def get_choice(self, data):
    return random.randint(0, self.choice_count)
