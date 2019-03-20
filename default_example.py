from data_generator import ClickGenerator
from visitor_model import VisitorModel

types = [(0,0), (0,1), (1,0), (1,1)]


def default_sex_age_click_gen():
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
  return click_gen

def test_default_sex_age_click_gen():
  click_gen = default_sex_age_click_gen()
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


def get_batches(batch_size, batch_count):
  feature_names = ['sex', 'age']
  visit_params = [0.5, 0.5]
  return get_visitor_batches(batch_size, batch_count, feature_names, visit_params)

def get_visitor_batches(batch_size, batch_count, param_names, visit_params):
  # means expect 30% of visitors sex[0] and 60% placebo[0]
  vModel = VisitorModel(param_names, visit_params)
  return [vModel.generate_visits(batch_size) for i in range(batch_count)]

