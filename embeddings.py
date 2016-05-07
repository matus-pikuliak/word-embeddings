from libraries import *
import svm_helper as svm
import data_helper as data


def load_model(seed_file=None):
    """
    Loads model of language as set of vectors. It is filtering the loaded vectors with top100k set containing
    words that should be loaded. Words from seed_File (file with exemplary pairs) are added to top100k permanently.
    :param seed_file: string
    :return: Word2Vec model
    """
    if seed_file is not None:
        add_file_to_top100k(seed_file)
    model = Word2Vec.load_word2vec_format(config.word2vec_file, binary=True, selected_words=top100k())
    print "Model successfully loaded"
    return model


"""
Creates global variable 'model' used in other classes.
"""
model = load_model()


def embedding_object(word=None, vector=None):
    """
    This method serves as a interface to embedding cache. If the embedding with given word was already
    used it will return this object. Otherwise it will create new object with specified vector.
    :param word: string
    :param vector: list of floats
    :return: Embedding
    """
    if word is None:
        return Embedding(vector=vector)

    cached = data.cached_embedding(word)
    if cached is None:
        data.embeddings[word] = Embedding(word=word, vector=vector)
    return data.embeddings[word]


class Embedding:

    """
    This class represents one word embedding as word and the vector it is assigned. These objects are usually
    not created directly but only through embedding_object() interface.
    """

    def __init__(self, word=None, vector=None):
        """
        :param word: string
        :param vector: list of floats
        """
        if vector is None and word is None:
            raise KeyError('You have to state word or vector')
        self.v = model[word] if vector is None else vector
        self.word = word

    def __len__(self):
        """
        Length of vector
        :return: integer
        """
        return len(self.v)

    def __sub__(self, embedding):
        """
        Calculates substraction of self and given embedding as new Embedding. Also uses embedding_object interface.
        :param embedding: Embedding
        :return: Embedding
        """
        if len(self) != len(embedding):
            raise KeyError('The embeddings have different lengths')
        vector = [self.v[i] - embedding.v[i] for i in xrange(len(self))]
        name = "%s-%s" % (embedding.word, self.word)
        return embedding_object(vector=vector, word=name)

    def neighbours(self, size=100):
        """
        Finds of given word neighbours with size n in vector space. Returns list of words.
        :param size: integer
        :return: list of strings
        """
        return [embedding_object(record[0]) for record in model.most_similar(self.word, topn=size)]


class Pair:

    """
    This class represents pair of embeddings forming one pair with semantic relation. Objects of this class can
    calculate similarity between themselves and can also be transformed to SVM-format string.
    """

    def __init__(self, embedding_1, embedding_2, positive=False, candidate=False):
        """
        Pair is created from two embeddings (1 & 2). positive and candidate are boolean variables indicating
        status of pair. At least one of them should be False.
        :param embedding_1, embedding_2: Embedding
        :param positive: True or False
        :param candidate: True or False
        """
        self.e_1 = embedding_1
        self.e_2 = embedding_2
        self.pair_embedding = embedding_2 - embedding_1
        self.positive = positive
        self.candidate = candidate

    def __len__(self):
        """
        Length of pair embedding vector
        :return: integer
        """
        return len(self.pair_embedding)

    def word(self):
        """
        Word assigned to given pair, it has a format of word1-word2, e.g. "Paris-France"
        :return: string
        """
        return self.pair_embedding.word

    def cosine_similarity(self, pair):
        """
        Cosine similarity between self and some other pair. It is using data heler to cache results.
        :param pair: Pair
        :return: float
        """
        return data.cosine_similarity(self.pair_embedding, pair.pair_embedding)

    def euclidean_similarity(self, pair):
        """
        Euclidean similarity between self and some other pair. It is using data heler to cache results.
        :param pair: Pair
        :return: float
        """
        return data.euclidean_similarity(self.pair_embedding, pair.pair_embedding)

    def neighbours(self, size=100):
        """
        Generates candidates for this given pair as product of neighborhood of its two embeddings.
        :return: list of Pairs
        """
        ng_1 = self.e_1.neighbours(size)
        ng_2 = self.e_2.neighbours(size)
        return [Pair(e_1, e_2, candidate=True) for e_1 in ng_1 for e_2 in ng_2 if e_1 != e_2]

    def svm_transform(self, pairs, distance='euclidean'):
        """
        Transforms given pairs to SVM sample with values calculated as similarities to given pairs with given distance.
        :param pairs: list of Pairs
        :param distance: 'euclidean' or 'cosine'
        :return: string in SVM sample format. See README for details.
        """
        label = self.svm_label()
        values = self.svm_values(pairs, distance)
        comment = self.word()
        return '%d %s # %s' % (label, values, comment)

    def svm_values(self, pairs, distance):
        """
        Creates the values for SVM sample. These values are the similarities themselves numbered from 1 upwards.
        E.g. "1:0.255 2:0.133 3:0.445" etc
        :param pairs: list of Pairs
        :param distance: 'euclidean' or 'cosine'
        :return: string in SVM sample values format. See README for details.
        """
        if distance == 'euclidean':
            vec = [self.euclidean_similarity(pair) for pair in pairs]
        if distance == 'cosine':
            vec = [self.cosine_similarity(pair) for pair in pairs]
        return svm.svm_transform_vector(vec)

    def svm_label(self):
        """
        Generates the SVM label for pair. This label is 1 for positive and -1 for unlabeled samples
        :return: 1 or -1
        """
        if self.positive:
            return 1
        if self.candidate:
            return -1
        raise KeyError('Relation should be marked candidate or positive.')


class PairSet:
    
    """
    This class represents set of pairs, usually seed set used to generate new pairs. It can calculate various metrics
    concerning such sets as well as generate new pairs in accordance with our method.
    """

    def __init__(self, pairs, filename=None):
        """
        filename is a path to file, if the set was created from one.
        :param pairs: list of Pairs 
        :param filename: string
        """
        self.set_pairs = pairs
        self.filename = filename

    @classmethod
    def create_from_file(cls, filename):
        """
        Takes files in our input format (see README) and transforms it into PairSet object.
        :param filename: string
        :return: 
        """
        with open(filename) as f:
            pairs = [line.strip().split() for line in f]
            seed_pairs = [Pair(embedding_object(pair[0]), embedding_object(pair[1]), positive=True) for pair in pairs]
            return PairSet(seed_pairs, filename=filename)

    def __len__(self):
        """
        Number of pairs in set.
        :return: integer
        """
        return len(self.set_pairs)

    def spatial_candidates(self, size=100, filter_positives=True):
        """
        Generates candidates in accordance to the proposal of our method. Candidates are calculates as a union
        of candidates generated from individual pairs (see Pair.spatial_candidates()). Candidates
        identical with positive pairs are removed if filter_positives is True.
        :param size: integer
        :param filter_positives: True or False
        :return: list of Pairs
        """
        all_candidates = flatten([pair.neighbours(size) for pair in self.set_pairs])

        # Embeddings of positive pairs are added so they don't end up in final set of candidates if filter_positives is True
        keys = set([pair.pair_embedding for pair in self.set_pairs]) if filter_positives else set()

        # Candidates are filtered so each pair is unique.
        candidates = list()
        for candidate in all_candidates:
            if candidate.pair_embedding not in keys:
                keys.add(candidate.pair_embedding)
                candidates.append(candidate)
        return candidates

    def spatial_candidates_words(self, size=100, filter_positives=True):
        """
        Faster way of generating candidates. Instead of Pairs objects only string with the name
        of given pair are generated (e.g. 'Paris-France'). Candidates
        identical with positive pairs are removed if filter_positives is True.
        :param size: integer
        :return: list of strings
        """
        candidates_words = set()
        for rel in self.set_pairs:
            ng_1 = [record[0] for record in model.most_similar(rel.e_1.word, topn=size)]
            ng_2 = [record[0] for record in model.most_similar(rel.e_2.word, topn=size)]
            for word in ["%s-%s" % (e_1, e_2) for e_1 in ng_1 for e_2 in ng_2 if e_1 != e_2]:
                candidates_words.add(word)
            if filter_positives:
                for word in ["%s-%s" % (pair.e_1.word, pair.e_2.word) for pair in self.set_pairs]:
                    if word in candidates_words:
                        candidates_words.remove(word)
        return candidates_words

    def spatial_candidates_size(self, size=100):
        """
        Quickly returns number of candidates generated from Set.
        :param size: integer
        :return: integer
        """
        return len(self.spatial_candidates_words(size))

    def pair_weight(self, pair, distance='euclidean'):
        """
        Calculates the weight of Pair in PairSet as an average of similarities to the rest of the pairs in Set.
        :param pair: Pair
        :param distance: 'euclidean' or 'cosine'
        :return: float
        """
        if distance == 'euclidean':
            similarities = [pair.euclidean_similarity(other_pair) for other_pair in self.set_pairs]
        elif distance == 'cosine':
            similarities = [pair.cosine_similarity(other_pair) for other_pair in self.set_pairs]
        else:
            raise KeyError('Wrong distance parameter')

        # -1 and -1 because the pair itself is in self.set_pairs skewing the results
        return (sum(similarities) - 1) / (len(similarities) - 1)

    def comparative_algorithm(self,
                              seed=None,
                              testing=None,
                              candidates=None,
                              method='avg',
                              weight_type='none',
                              distance='euclidean',
                              **kwargs):
        """
        Calculates comparative algorithms from out method on given data. It rates list of candidates based on their
        similarity to seed set. If seed is not set, self is the seed. If candidates are not set,
        they are generated from seed. If testing is set, it is added to candidates as positive samples
        to control how they are rated. Method is returning ResultList object so it can be later printed or evaluated.
        Method is 'max' or 'avg' saying how are similarities with seed set handled.
        Weight_type is 'none ', 'normalize' or 'softmax' and it says how are similarities normalizes.
        Distance is 'cosine' or 'euclidean' and it says what kind of similarity is used in computations.
        :param seed:        PairSet
        :param testing:     PairSet
        :param candidates:  list of Pairs
        :param method:      'avg' or 'max'
        :param weight_type: 'none', 'normalized' or 'softmax'
        :param distance:    'cosine' or 'euclidean'
        :param kwargs:      ...
        :return:            rResultList
        """

        if seed is None:
            seed = self
        if candidates is None:
            candidates = seed.spatial_candidates()

        results = ResultList()
        evaluated = list(candidates)
        if testing is not None:
            evaluated += testing.set_pairs

        if weight_type == 'none':
            weights = [float(1)/len(seed) for _ in xrange(len(seed))]
        elif weight_type == 'normalized':
            weights = normalize_list([seed.pair_weight(pair, distance) for pair in seed.set_pairs])
        elif weight_type == 'softmax':
            weights = softmax_list([seed.pair_weight(pair, distance) for pair in seed.set_pairs])
        else:
            raise KeyError('Wrong weight_type parameter')

        for pair in evaluated:
            positive = pair.positive
            name = pair.word()
            if distance == 'cosine':
                similarities = [pair.cosine_similarity(seed_pair) for seed_pair in seed.set_pairs]
            elif distance == 'euclidean':
                similarities = [pair.euclidean_similarity(seed_pair) for seed_pair in seed.set_pairs]
            else:
                raise KeyError('Wrong distance parameter')

            if method == 'max':
                final_similarity = max(similarities)
            elif method == 'avg':
                final_similarity = sum([
                        weights[i] * similarities[i]
                        for i in xrange(len(similarities))
                    ])
            else:
                raise KeyError('Wrong method parameter')

            results.append(is_positive=positive,name=name,score=final_similarity)
        return results

    def pu_learning(self,
                    seed=None,
                    testing=None,
                    candidates=None,
                    distance='euclidean',
                    **kwargs):
        """
        Calculates PU Learning algorithms from out method on given data. If seed is not set, it is calculated over
        self. Seed is used as a set of pairs used to rate other pairs. If candidates are not set,
        they are generated from seed. If testing is set, it is added to candidates as positive samples
        to control how they are rated. Method is returning ResultList object so it can be later printed or evaluated.
        Distance is 'cosine' or 'euclidean' and it says what kind of similarity is used in computations.

        SVM files are created while calculating this method. Each set of SVM files has unique timestamp that ties them together.
        :param seed:        PairSet
        :param testing:     PairSet
        :param candidates:  list of Pairs
        :param distance:    'cosine' or 'euclidean'
        :param kwargs:      ...
        :return:            ResultList
        """
        if seed is None:
            seed = self
        if candidates is None:
            candidates = seed.spatial_candidates()
        evaluated = list(candidates)
        if testing is not None:
            evaluated += testing.set_pairs
        candidates = self.spatial_candidates()
        examples = seed.set_pairs[0::2]
        positive = seed.set_pairs[1::2]

        name = re.match('.*/([a-z]*).txt',self.filename).group(1)
        timestamp = str(int(time.time()))
        print "SVM timestamp is %s" % timestamp
        train_filename = svm.svm_file_name(name, timestamp, 'train')
        model_filename = svm.svm_file_name(name, timestamp, 'model')
        test_filename = svm.svm_file_name(name, timestamp, 'test')
        prediction_filename = svm.svm_file_name(name, timestamp, 'prediction')

        open(train_filename, "wb").write('\n'.join([rel.svm_transform(examples, distance) for rel in positive + candidates]))
        open(test_filename, "wb").write('\n'.join([rel.svm_transform(examples, distance) for rel in evaluated]))
        os.system('./lib/svm_perf/svm_perf_learn -v 0 -l 10 -c 0.01 -w 3 %s %s' % (train_filename, model_filename))
        os.system('./lib/svm_perf/svm_perf_classify -v 0 %s %s %s' % (test_filename, model_filename, prediction_filename))

        return svm.svm_timestamp_to_results(timestamp)

    def find_new_pairs(self, n=100, filename=config.default_output_file, **kwargs):
        """
        Finds new pairs from given set and prints them to file. N is number of results printed to file called filename.
        kwargs must contain method attribute with values 'max', 'avg' or 'pu'. Details about other attributes in kwargs
        can bee seen in pu_learning() and comparative_algorithm() methods.
        :param n: integer
        :param filename: string
        :param kwargs: algorithm parameters
        :return: None
        """
        if kwargs['method'] == 'max' or kwargs['method'] == 'avg':
            results = self.comparative_algorithm(**kwargs)
        if kwargs['method'] == 'pu':
            results = self.pu_learning(**kwargs)
        results.print_top_n_to_file(n, filename)

    def seed_recall(self, size=100, interesting_pairs=None):
        """
        Calculates the seed recall metric of given set. Details about this metric is in the proposal of our method.
        interesting_relations can be used to check for other pair than the pairs in our set. This can be used
        in different experiments. Size can be used to control size of neighborhood used for generating candidates.
        :param size: integer
        :param interesting_pairs: list of Pairs
        :return: float
        """
        if interesting_pairs is None:
            interesting_pairs = self.set_pairs
        candidates_words = self.spatial_candidates_words(size, filter_positives=False)
        present = len([1 for pair in interesting_pairs if pair.word() in candidates_words])
        return float(present) / len(interesting_pairs)

    def positions_testing(self, repeat=10, **kwargs):
        """
        Tests positions on which the positive samples are rated in sorted set of candidates. Details about this metric
        is in the proposal of our method. Repeat says how many times should the experiment repeat, the more the more
        precise are the results. However this experiment is very timely. Kwargs must contain method attribute
        with value 'avg', 'max' or 'pu'. Details about other attributes in kwargs
        can bee seen in pu_learning() and comparative_algorithm() methods.
        :param repeat: integer
        :param kwargs: algorithm parameters
        :return: None
        """
        positions = []
        candidates = self.spatial_candidates()
        for _ in xrange(repeat):
            testing, training = self.testing_and_training_set(0.2)
            if kwargs['method'] == 'max' or kwargs['method'] == 'avg':
                results = self.comparative_algorithm(seed=training, testing=testing, candidates=candidates, **kwargs)
            if kwargs['method'] == 'pu':
                results = self.pu_learning(seed=training, testing=testing, candidates=candidates, **kwargs)
            positions.append(results.positive_positions())
        positions = flatten(positions)
        print_vector_stats(positions)
        return np.median(positions)

    def testing_and_training_set(self, testing_proportion):
        """
        Split the set's relations into testing and training set.
        testing_proportion is indicating what proportion (0.2 means 20% e.g.)
        of the original set l will be used as testing items.
        :param testing_proportion: float <0,1>
        :return: tuple of two sets
        """
        size = int(len(self) * testing_proportion)
        if size < 1 or (len(self) - size) < 1:
            raise AttributeError('One of the sets has non-positive size.')
        shuffled = list(self.set_pairs)
        random.shuffle(shuffled)
        return PairSet(shuffled[0:size]), PairSet(shuffled[size:])


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
        See README for more details.
        :param n: integer
        :param filename: string
        :return: None
        """
        self.sort()
        with open(filename, 'w+') as f:
            for i in xrange(n):
                f.write("?\t%s\n" % self[i].name)


# for f in glob.glob('./relations/capitals.txt'):
#     our_set = PairSet.create_from_file(f)
#     our_set.positions_testing(method='pu')
#     our_set.positions_testing(method='pu', distance='cosine')
#     our_set.positions_testing(method='max')
#     our_set.positions_testing(method='avg')
#     our_set.positions_testing(method='avg', distance='cosine')
#     our_set.positions_testing(method='avg', weight_type='softmax')
#     data.clear_cache()
#
# for f in glob.glob('./relations/currency.txt'):
#     our_set = PairSet.create_from_file(f)
#     our_set.positions_testing(method='pu')
#     our_set.positions_testing(method='pu', distance='cosine')
#     our_set.positions_testing(method='max')
#     our_set.positions_testing(method='avg')
#     our_set.positions_testing(method='avg', distance='cosine')
#     our_set.positions_testing(method='avg', weight_type='softmax')
#     data.clear_cache()


for f in glob.glob('./relations/cities.txt'):
    our_set = PairSet.create_from_file(f)
    our_set.positions_testing(method='max')
    our_set.positions_testing(method='avg')
    our_set.positions_testing(method='pu')
    our_set.positions_testing(method='pu', distance='cosine')
    our_set.positions_testing(method='avg', distance='cosine')
    our_set.positions_testing(method='avg', weight_type='softmax')
    data.clear_cache()

for f in glob.glob('./relations/family.txt'):
    our_set = PairSet.create_from_file(f)
    our_set.positions_testing(method='max')
    our_set.positions_testing(method='avg')
    our_set.positions_testing(method='pu')
    our_set.positions_testing(method='pu', distance='cosine')
    our_set.positions_testing(method='avg', distance='cosine')
    our_set.positions_testing(method='avg', weight_type='softmax')
    data.clear_cache()


    # exit()
    # our_set.find_new_pairs(method='avg', distance='euclidean', weight_type='none', filename='part_euc.txt')
    # our_set.find_new_pairs(method='avg', distance='cosine', weight_type='none', filename='part_cos.txt')
    # data.clear_cache()
    # exit()
    # #our_set.run_sim_test()
    # candidates = our_set.spatial_candidates()
    # relations = our_set.relations
    # for i in xrange(10):
    #     i = i+1
    #     results = list()
    #     for j in xrange(1):
    #         random.shuffle(relations)
    #         training = RelationSet(relations[0:i])
    #         testing = RelationSet(relations[-5:])
    #         print len(training), len(testing)
    #         positions = our_set.sim_measure(training, testing, candidates, distance='euclidean', weight_type='none')
    #         results.append(positions)
    #     print i, print_vector_stats(flatten(results))