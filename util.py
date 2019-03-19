def print_clicks(types, batch_count, num_choices, clicks, visits, choices):
  counts = []
  for i in range(num_choices):
    type_counts = { t:[0,0] for t in types}
    counts.append(type_counts)
  for choice, visit, click in zip(choices, visits, clicks):
    key = tuple(visit)
    counts[choice][key][click] += 1
  for i, c in enumerate(counts):
    print("Batch", batch_count, " Choice: ", i, " clicks")
    for k,v in c.items():
      pct = v[1]/(v[0]+v[1]) if v[0]+v[1] > 0 else 0.0
      print(k, v, '{:.1%}'.format(pct))
  print("---")

