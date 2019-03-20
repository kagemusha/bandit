from collections import defaultdict
import numpy as np

def print_clicks(types, batch_count, clicks, visits, choices):
  type_counts = { t:[0,0] for t in types}
  counts = defaultdict(lambda: type_counts)
  for choice, visit, click in zip(choices, visits, clicks):
    key = tuple(visit)
    counts[choice][key][click] += 1

  choices = sorted(counts.keys())
  for choice in choices:
    print("print_clicks>> Batch", batch_count, " Choice: ", choice)
    ch_counts = counts[choice]
    for k,v in ch_counts.items():
      pct = v[1]/(v[0]+v[1]) if v[0]+v[1] > 0 else 0.0
      print(k, v, '{:.1%}'.format(pct))
  print("---")

def __main__():
  types = [(0,0), (0,1), (1,0), (1,1)]
  n = 30
  clicks = np.random.choice(range(2), n)
  choices = np.random.choice(range(1), n)
  visits = [types[i] for i in np.random.choice(len(types), n)]
  print_clicks(types, 1, clicks, visits, choices)

if __name__ == "__main__":
  __main__()
