# file IO helpers
import pickle


def save_pickle(file, file_path):
    with open(file_path, "wb") as fp:
        pickle.dump(file, fp)


def load_pickle(file_path):
    with open(file_path, "rb") as fp:
        x = pickle.load(fp)
    return x
