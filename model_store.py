import pickle

MODEL_PREFIX = "_htmodel"

def get_model_filename(choice):
  return f"{MODEL_PREFIX}_{choice}"

def save_model_file(choice, model):
  filename = get_model_filename(choice)
  print("pickling ", filename)
  pickle_model(filename, model)

def pickle_model(filename, model):
  outfile = open(filename, 'wb')
  pickle.dump(model, outfile)
  outfile.close()

def load_model_file(choice):
  filename = get_model_filename(choice)
  print("unpickling ", filename)
  return load_model(filename)

def load_model(filename):
  infile = open(filename, 'rb')
  model = pickle.load(infile)
  infile.close()
  return model
