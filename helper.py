from libraries import *

def flatten(l):
    """
    Takes all the elements from list l and extracts them to returned array
    :param l: list of lists
    :return: list
    """
    return [item for sublist in l for item in sublist]


def softmax_list(l):
    """
    Normalizes array l with softmax function
    :param l: list of numbers
    :return: list
    """
    return np.exp(l) / np.sum(np.exp(l), axis=0)


def normalize_list(l):
    """
    Normalizes array l with softmax function
    :param l: list of numbers
    :return: list
    """
    return [item / sum(l) for item in l]


def print_vector_stats(vec, verbose=True):
    """
    Prints the statistics - median, std, mean - about given vector
    Verbose option prints names of metrics aswell.
    :param vec: list of numbers
    :param verbose: True or False
    :return: None
    """
    median = float(np.median(vec))
    std = float(np.std(vec))
    mean = float(np.mean(vec))
    output = (median, std, mean)
    if verbose:
        print "median: %f\tstd: %f\tmean: %f" % output
    else:
        print "%f\t%f\t%f" % output