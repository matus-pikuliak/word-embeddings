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


def top100k():
    """
    Processes the file with selected words. The file has one word on each line.
    Lines starting with '#' are considered comments.
    :return: set of strings
    """
    top100k_set = set()
    with open('./wiki-100k.txt') as f:
        for line in f:
            if not line.startswith('#'):
                top100k_set.add(line.strip())
    return top100k_set


def add_word_to_top100k(word):
    with open('./wiki-100k.txt', 'a') as f:
        f.write("\n" + word)


def add_file_to_top100k(filename):
    top100k_set = top100k()
    with open(filename) as f:
        for line in f:
            for word in line.strip().split():
                if word not in top100k_set:
                    add_word_to_top100k(word)

