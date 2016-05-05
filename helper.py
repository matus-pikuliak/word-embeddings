from libraries import *

def flatten(l):
    """
    Takes all the elements from list l and extract them to returned array
    :param l: list of lists
    :return: list
    """
    return [item for sublist in l for item in sublist]


def softmax_list(l):
    """
    Normalize array l with softmax function
    :param l: list of numbers
    :return: list
    """
    return np.exp(l) / np.sum(np.exp(l), axis=0)


def normalize_list(l):
    """
    Normalize array l with softmax function
    :param l: list of numbers
    :return: list
    """
    return [x / sum(l) for x in l]