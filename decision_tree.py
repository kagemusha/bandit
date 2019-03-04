import random

def test_split(index, value, dataset):
  left, right = list(), list()
  for row in dataset:
    if row[index] < value:
      left.append(row)
    else:
      right.append(row)
  return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
  # count all samples at split point
  n_instances = float(sum([len(group) for group in groups]))
  # sum weighted Gini index for each group
  gini = 0.0
  for group in groups:
    size = float(len(group))
    # avoid divide by zero
    if size == 0:
      continue
    score = 0.0
    # score the group based on the score for each class
    for class_val in classes:
      p = [row[-1] for row in group].count(class_val) / size
      score += p * p
    # weight the group score by its relative size
    gini += (1.0 - score) * (size / n_instances)
  return gini


# Select the best split point for a dataset
def get_split(dataset):
  class_values = list(set(row[-1] for row in dataset))
  print("CVs:", class_values)
  b_index, b_value, b_score, b_groups = 999, 999, 999, None
  for index in range(len(dataset[0]) - 1):
    group_vals = list(set(row[index] for row in dataset))
    group_vals = sorted(group_vals)[1:]
    print("gv:", index, group_vals)
    for val in group_vals:
      groups = test_split(index, val, dataset)
      gini = gini_index(groups, class_values)
      print('X%d < %.3f Gini=%.6f' % ((index + 1), val, gini))
      if gini < b_score:
        b_index, b_value, b_score, b_groups = index, val, gini, groups
  return {'index': b_index, 'value': b_value, 'groups': b_groups}


def sim1():
  dataset = [[2.771244718, 1.784783929, 4, 0],
             [1.728571309, 1.169761413, 3, 0],
             [3.678319846, 2.81281357, 2, 0],
             [3.961043357, 2.61995032, 17, 0],
             [2.999208922, 2.209014212, 5, 0],
             [7.497545867, 3.162953546, 15, 1],
             [9.00220326, 3.339047188, 3, 1],
             [7.444542326, 0.476683375, 6, 1],
             [10.12493903, 3.234550982, 9, 1],
             [6.642287351, 3.319983761, 17, 1]]

  split = get_split(dataset)
  print('Split: [X%d < %.3f]' % ((split['index'] + 1), split['value']))

# sim1()


def simulate_data(count):
  # flist = ['gender','placebo']
  malePct = .4
  placeboPct = .6
  maleClickPcts = [0.05, 0.04]
  femaleClickPcts = [0.04, 0.05]
  ds = []
  for i in range(count):
    sex = 1 if random.random() < malePct else 0
    p = 1 if random.random() < placeboPct else 0
    choice = random.randint(1,2)
    clickPcts = maleClickPcts if sex == 1 else femaleClickPcts
    propensity = clickPcts[choice-1]
    click = 1 if random.random() < propensity else 0
    ds.append([sex,p,choice,click])
  return ds


def print_split(data, choice):
  split = get_split(data)
  print('Split: [X%d < %.3f]' % ((split['index'] + 1), split['value']))
  g0 = split['groups'][0]
  g1 = split['groups'][1]
  clicksG0 = sum([obs[3] for obs in g0])
  clicksG1 = sum([obs[3] for obs in g1])
  print("Choice: ", choice)
  print(f"G0 clicks/tot:  {clicksG0}/{len(g0)} G1 clicks: {clicksG1}/{len(g1)}")
  print("G0 click pct: ", clicksG0/len(g0), "G1 clicks: ", clicksG1/len(g1))


c1, c2 = simulate_data(100000)
print_split(c1, 1)
print_split(c2, 2)
