from libraries import *

class ResultFile:

    """
    This class is an interface to result files. See README for details about these files.
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



