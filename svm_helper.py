from libraries import *
import results_helper as res


def svm_file_name(set_name, timestamp, file_type):
    """
    Creates filename for SVM file. set_name is name of given seed set, e.g. capitals.
    timestamp is an time identifier of created sef of SVM files. file_type is a type
    of SVM file created (train, test, model, prediction)
    :param set_name: string
    :param timestamp: integer
    :param file_type: string
    :return: string
    """
    return "%s%s-%s-%s-svm" % (config.svm_folder, set_name, timestamp, file_type)


def svm_timestamp_to_results(timestamp):
    """
    Finds SVM files created for given timestamp and transforms them to ResultList object.
    :param timestamp: string
    :return: ResultList
    """

    try:
        test_file = glob.glob("%s*%s-test*" % (config.svm_folder, timestamp))[0]
        predict_file = glob.glob("%s*%s-prediction*" % (config.svm_folder, timestamp))[0]
    except IndexError:
        raise AttributeError('Given timestamp does not have files generated')

    results = res.ResultList()
    with open(test_file) as f:
        for line in f:
            positive = (line.split()[0] == '1')
            name = line.split('#')[1].strip()
            results.append(is_positive=positive, name=name)
    with open(predict_file) as f:
        i = 0
        for line in f:
            results[i].score = float(line)
            i += 1
    return results


def svm_filename_to_timestamp(filename):
    """
    Extracts timestamp from given filename.
    :param filename: string
    :return: string
    """
    try:
        match = re.match(('%s[a-z]*-([0-9]*).*' % config.svm_folder), filename).group(1)
    except AttributeError:
        raise AttributeError('Given filename does not have correct name')
    return match


def svm_set_positions(seed_set_name):
    """
    Finds all the SVM files sets with given seed set name, transforms them to ResultList and calculates
    statistics over positive positions of these results.
    :param seed_set_name: string
    :return: None
    """
    files = glob.glob('%s%s*prediction*' % (config.svm_folder, seed_set_name))
    timestamps = [svm_filename_to_timestamp(filename) for filename in files]
    positions = flatten([svm_timestamp_to_results(timestamp).positive_positions() for timestamp in timestamps])
    print_vector_stats(positions)


def svm_transform_vector(vec):
    """
    Transforms vector vec into SVM vector format:
    E.g.: 1:value_1 2:value_2 3:value_3 etc.
    :param vec: list
    :return: string
    """
    return ' '.join(["%d:%f" % (i+1, vec[i]) for i in xrange(len(vec))])