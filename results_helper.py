from libraries import *

class Result:

    """
    One result from our experiment consisting of pair, its score, position
    and information about whether it is positive or not.
    """

    def __init__(self, name=None, is_positive=None, score=None, position=None):
        """
        name is the pairs of words from given pair e.g. "Paris France".
        is_positive says whether the result is from positive sample or not (candidate).
        score says how was given pair scored by scoring algorithm.
        position says what is the position of result in given result list.
        :param name: string
        :param is_positive: True or False
        :param score: float
        :param position: integer
        """
        self.name = name
        self.is_positive = is_positive
        self.score = score
        self.position = position


class ResultList:

    """
    List of Results from given experiment. It contains numerous Results and it is capable of working them,
    e.g. sorting them and printing them.
    """

    def __init__(self):
        self.results_list = []

    def __len__(self):
        return len(self.results_list)

    def __getitem__(self, item):
        return self.results_list[item]

    def append(self, **kwargs):
        """
        Appends new result to list of results. Allowed arguments are listed in Result class definition.
        :param kwargs: See Result class
        :return: None
        """
        result = Result(**kwargs)
        self.results_list.append(result)

    def sort(self):
        """
        Sorts results in result list and assigns them positions starting with 1.
        :return: None
        """
        self.results_list = sorted(self.results_list, key=lambda result: -result.score)
        for i in xrange(len(self)):
            self[i].position = i + 1

    def positive_positions(self):
        """
        Returns the list of positions occupied by positive samples. When calculating position
        for one positive sample, all the other positive samples are virtually removed from the list.
        The position is therefore calculated only compared to negative samples.

        Example: [positive, negative, positive] has positions [1,2] instead of [1,3] because while
        calculating position of third sample we ignore the first.

        :return: list of integers
        """
        self.sort()
        positions = [result.position for result in self.results_list if result.is_positive]
        for i in xrange(len(positions)):
            positions[i] -= i
        return positions

    def print_top_n(self, n=100):
        """
        Prints names of top n results in our result list.
        :param n: integer
        :return: None
        """
        self.sort()
        for i in xrange(n):
            print self[i].name

    def print_top_n_to_file(self, n, filename):
        """
        Prints names of top n results to file called filename in accordance with out result file format.
        See readme for more details.
        :param n: integer
        :param filename: string
        :return: None
        """
        self.sort()
        with open(filename, 'w+') as f:
            for i in xrange(n):
                f.write("?\t%s\n" % self[i].name)


class ResultFile:

    """
    This class is an interface to result files. See readme for details about these files.
    This class is capable of loading these files and calculating different statitics about them.
    """

    def __init__(self, filename):
        """
        Processes file with results. Each line is transformed to list with two values:
        1. starting letter 2. name of pair (e.g. ['r', 'Paris-France'])
        :param filename: string
        """
        self.records = []
        with open(filename, 'r') as f:
            for line in f:
                self.records.append(line.strip().split('\t'))

    def __len__(self):
        return len(self.records)

    def is_correct(self, index):
        """
        Checks if i-th record in file is correct
        :param index: integer
        :return: True or False
        """
        return self.records[index][0].startswith('r')

    def is_partially_correct(self, index):
        """
        Checks if i-th record in file is partially correct
        :param index: integer
        :return: True or False
        """
        return self.records[index][0].startswith('p')

    def is_incorrect(self, index):
        """
        Checks if i-th record in file is incorrect
        :param index: integer
        :return: True or False
        """
        return self.records[index][0].startswith('w')

    def correct_count(self):
        """
        Counts how many records in file are correct
        :return: integer
        """
        return len([i for i in xrange(len(self)) if self.is_correct(i)])

    def partially_correct_count(self):
        """
        Counts how many records in file are partially correct
        :return: integer
        """
        return len([i for i in xrange(len(self)) if self.is_partially_correct(i)])

    def incorrect_count(self):
        """
        Counts how many records in file are incorrect
        :return: integer
        """
        return len([i for i in xrange(len(self)) if self.is_incorrect(i)])

    @staticmethod
    def ndcg_score(relevancy, position):
        """
        Calculates NDCG score for given position and relevancy of element.
        :param relevancy: float
        :param position: integer
        :return: float
        """
        return relevancy / math.log(position+1, 2)

    def idcg(self):
        """
        Calculates ideal DCG score for virtual set of the same size.
        :return: float
        """
        return sum([self.ndcg_score(1,pos+1) for pos in xrange(len(self))])

    def dcg(self):
        """
        Calculates DCG score of our set of records. Relevancy for our results are:
        correct ............ 1
        partially correct .. 0.5
        incorrect .......... 0
        :return: float
        """
        scores = []
        for i in xrange(len(self)):
            relevancy = 0
            if self.is_correct(i):
                relevancy = 1
            if self.is_partially_correct(i):
                relevancy = 0.5
            scores.append(self.ndcg_score(relevancy, i+1))
        return sum(scores)

    def ndcg(self):
        """
        Calculates NDCG for our set of recrords
        :return: float
        """
        return self.dcg() / self.idcg()

    def similarity(self, other_record_file):
        """
        Calculates similarity to other RecordFile as number of common pairs to total number of pairs
        :param other_record_file: RecordFile
        :return: float
        """
        words = [record[1] for record in self.records]
        common_words = [record[1] for record in other_record_file.records if record[1] in words]
        return float(len(common_words)) / len(self)

    def average_word_occurence_count(self):
        """
        Calculate how many times in average is one word repeating itself in results.
        :return: float
        """
        words = flatten([record[1].split('-') for record in self.records])
        print np.mean(Counter(words).values())

    def correct_at_k(self):
        """
        Return array stating how many correct pairs are in first k records.
        This is calculated for every k from 1 to len(self) and it is returned as list of tuples.
        E.g. tuple (5,2) means that in first 5 results were 2 correct or partially correct.
        :return: list of tuples
        """
        correct = 0
        results = []
        for i in xrange(len(self)):
            if self.is_correct(i) or self.is_partially_correct(i):
                correct += 1
            results.append((i+1, correct))
        return results



