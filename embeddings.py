from libraries import *
import results_helper as res
import svm_helper as svm
import data_helper as data

"""
model variable is containg word2vec model of natural language. The file containing the vectors is defined in config.
"""
model = Word2Vec.load_word2vec_format(config.word2vec_file, binary=True, selected_words=top100k())
print "Model successfully loaded"


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

    def spatial_candidates(self, size=100):
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
        all_candidates = flatten([pair.spatial_candidates(size) for pair in self.set_pairs])

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
                              method='average',
                              weight_type='softmax',
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
        :return:            res.ResultList
        """

        if seed is None:
            seed = self
        if candidates is None:
            candidates = seed.spatial_candidates()

        results = res.ResultList()
        evaluated = candidates
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
                    training=None,
                    testing=None,
                    candidates=None,
                    distance='euclidean',
                    **kwargs):
        """
        Calculates comparative algorithms from out method on given data. If seed is not set, it is calculated over
        self. Seed is used as a set of pairs used to rate other pairs. If candidates are not set,
        they are generated from seed. If testing is set, it is added to candidates as positive samples
        to control how they are rated. Method is returning ResultList object so it can be later printed or evaluated.
        Method is 'max' or 'avg' saying how are similarities with seed set handled.
        Weight_type is 'none ', 'normalize' or 'softmax' and it says how are similarities normalizes.
        Distance is 'cosine' or 'euclidean' and it says what kind of similarity is used in computations.
        :param seed:        PairSet
        :param testing:     PairSet
        :param candidates:  list of Pairs
        :param distance:    'cosine' or 'euclidean'
        :param kwargs:      ...
        :return:            res.ResultList
        """
        if training is None:
            training = self
        if candidates is None:
            candidates = training.spatial_candidates()
        evaluated = candidates
        if testing is not None:
            evaluated += testing.set_pairs
        candidates = self.spatial_candidates()
        examples = training.set_pairs[0::2]
        positive = training.set_pairs[1::2]

        name = re.match('.*/([a-z]*).txt',self.filename).group(1)
        timestamp = str(int(time.time()))
        print "SVM timestamp is %s" % timestamp
        train_filename = svm.svm_file_name(name, timestamp, 'train')
        model_filename = svm.svm_file_name(name, timestamp, 'model')
        test_filename = svm.svm_file_name(name, timestamp, 'test')
        prediction_filename = svm.svm_file_name(name, timestamp, 'prediction')

        open(train_filename, "wb").write('\n'.join([rel.svm_transform(examples, distance) for rel in positive + candidates]))
        open(test_filename, "wb").write('\n'.join([rel.svm_transform(examples, distance) for rel in evaluated]))
        os.system('./svm-perf/svm_perf_learn -l 10 -c 0.01 -w 3 %s %s' % (train_filename, model_filename))
        os.system('./svm-perf/svm_perf_classify %s %s %s' % (test_filename, model_filename, prediction_filename))

        return svm.svm_timestamp_to_results(timestamp)

    def find_new_pairs(self, n=100, **kwargs):
        if kwargs['method'] == 'max' or kwargs['method'] == 'avg':
            results = self.comparative_algorithm(**kwargs)
        if kwargs['method'] == 'pu':
            results = self.pu_learning(**kwargs)
        results.print_top_n_to_file(n, config.output_file)

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

    def get_positions(self, repeat=20, **kwargs):
        positions = []
        for _ in xrange(repeat):
            testing, training = self.testing_and_training_set(0.2)
            positions.append(self.sim_measure(training, testing, **kwargs))
        return flatten(positions)

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

for f in glob.glob('./relations/capitals.txt'):
    our_set = PairSet.create_from_file(f)
    print our_set.seed_recall()
    exit()
    #our_set.run_sim_test()
    candidates = our_set.spatial_candidates()
    relations = our_set.relations
    for i in xrange(1):
        i = i+1
        results = list()
        for j in xrange(1):
            random.shuffle(relations)
            training = RelationSet(relations[0:i])
            testing = RelationSet(relations[-5:])
            print len(training), len(testing)
            positions = our_set.sim_measure(training, testing, candidates, distance='euclidean', weight_type='none')
            results.append(positions)
        print i, print_vector_stats(flatten(results))