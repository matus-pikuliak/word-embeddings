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

    def print_top_n(self, n):
        """
        Prints names of top n results in our result list.
        :param n: integer
        :return: None
        """
        self.sort()
        for i in xrange(n):
            print self[i].name



