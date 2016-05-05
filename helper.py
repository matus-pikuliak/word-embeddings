from libraries import *

def flatten(l):
    """
    Take all the elements from list l and extract them to returned array
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
    return [item / sum(l) for item in l]


def split(l, split_proportion):
    """
    Split the list l into testing and training set. testing_proportion is indicating what proportion
    of the original set l will be used as testing items.
    :param l: list
    :param testing_proportion: float <0,1>
    :return: tuple of two lists
    """
    size = int(len(l) * split_proportion)
    if size < 1 or (len(l) - size) < 1:
        raise AttributeError('One of the sets has non-positive size.')
    shuffled = list(l)
    random.shuffle(shuffled)
    return shuffled[0:size], shuffled[size:]