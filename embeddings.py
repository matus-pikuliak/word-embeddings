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

    def neighbours(self, n=100):
        """
        Finds of given word neighbours with size n in vector space. Returns list of words.
        :param n: integer
        :return: list of strings
        """
        return [embedding_object(record[0]) for record in model.most_similar(self.word, topn=n)]


class Pair:

    """
    This class represents pair of embeddings forming one pair with semantic relation. Objects of this class can
    calculate similarity between themselves and can also be transformed to SVM-format string.
    """

    def __init__(self, embedding_1, embedding_2, positive=False, candidate=False):
        self.e_1 = embedding_1
        self.e_2 = embedding_2
        self.pair_embedding = embedding_2 - embedding_1
        self.positive = positive
        self.candidate = candidate

    def __len__(self):
        return len(self.pair_embedding)

    def word(self):
        return self.pair_embedding.word

    def cosine_similarity(self, pair):
        return data.cosine_similarity(self.pair_embedding, pair.pair_embedding)

    def euclidean_similarity(self, pair):
        return data.euclidean_similarity(self.pair_embedding, pair.pair_embedding)

    def spatial_candidates(self):
        ng_1 = self.e_1.neighbours()
        ng_2 = self.e_2.neighbours()
        return [Pair(e_1, e_2, candidate=True) for e_1 in ng_1 for e_2 in ng_2 if e_1 != e_2]

    def svm_transform(self, pairs, distance='euclidean'):
        label = self.svm_label()
        values = self.svm_values(pairs, distance)
        comment = self.word()
        return '%d %s # %s' % (label, values, comment)

    def svm_values(self, pairs, distance):
        if distance == 'euclidean':
            vec = [self.euclidean_similarity(pair) for pair in pairs]
        if distance == 'cosine':
            vec = [self.cosine_similarity(pair) for pair in pairs]
        return svm.svm_transform_vector(vec)

    def svm_label(self):
        if self.positive:
            return 1
        if self.candidate:
            return -1
        raise KeyError('Relation should be marked candidate or positive.')


class PairSet:

    def __init__(self, pairs, filename=None):
        self.set_pairs = pairs
        self.filename = filename

    @classmethod
    def create_from_file(cls, filename, capitalize=False):
        def create_pair_object(pair):
            return Pair(embedding_object(pair[0]), embedding_object(pair[1]), positive=True)

        def capitalize_line(pair):
            return map(str.capitalize, pair)

        with open(filename) as f:
            pairs = [line.strip().split() for line in f]
            if capitalize:
                pairs = map(capitalize_line, pairs)
            seed_pairs = map(create_pair_object, pairs)
            return PairSet(seed_pairs, filename=filename)

    def __len__(self):
        return len(self.set_pairs)

    def spatial_candidates(self, filter_positives=True):
        all_candidates = flatten([pair.spatial_candidates() for pair in self.set_pairs])
        keys = set([pair.pair_embedding for pair in self.set_pairs]) if filter_positives else set()
        candidates = list()
        for candidate in all_candidates:
            if candidate.pair_embedding not in keys:
                keys.add(candidate.pair_embedding)
                candidates.append(candidate)
        return candidates

    def spatial_candidates_size(self, size=100):
        candidates_words = set()
        for rel in self.set_pairs:
            ng_1 = [record[0] for record in model.most_similar(rel.e_1.word, topn=size)]
            ng_2 = [record[0] for record in model.most_similar(rel.e_2.word, topn=size)]
            for word in ["%s-%s" % (e_1, e_2) for e_1 in ng_1 for e_2 in ng_2 if e_1 != e_2]:
                candidates_words.add(word)
            for word in ["%s-%s" % (pair.e_1.word, pair.e_2.word) for pair in self.set_pairs]:
                if word in candidates_words:
                    candidates_words.remove(word)
        return len(candidates_words)

    def pair_weight(self, pair, distance='euclidean'):
        if distance == 'euclidean':
            similarities = [pair.euclidean_similarity(other_pair) for other_pair in self.set_pairs]
        elif distance == 'cosine':
            similarities = [pair.cosine_similarity(other_pair) for other_pair in self.set_pairs]
        else:
            raise KeyError('Wrong distance parameter')
        return (sum(similarities) - 1) / (len(similarities) - 1)

    def comparative_algorithm(self, training=None, testing=None, candidates=None, method='average',
                              weight_type='softmax', distance='euclidean', **kwargs):

        if training is None:
            training = self
        if candidates is None:
            candidates = training.spatial_candidates()

        results = res.ResultList()
        evaluated = candidates
        if testing is not None:
            evaluated += testing.set_pairs

        if weight_type == 'none':
            weights = [float(1)/len(training) for _ in xrange(len(training))]
        elif weight_type == 'normalized':
            weights = normalize_list([training.pair_weight(pair, distance) for pair in training.set_pairs])
        elif weight_type == 'softmax':
            weights = softmax_list([training.pair_weight(pair, distance) for pair in training.set_pairs])
        else:
            raise KeyError('Wrong weight_type parameter')

        for pair in evaluated:
            positive = pair.positive
            name = pair.word()
            if distance == 'cosine':
                similarities = [pair.cosine_similarity(training_pair) for training_pair in training.set_pairs]
            elif distance == 'euclidean':
                similarities = [pair.euclidean_similarity(training_pair) for training_pair in training.set_pairs]
            else:
                raise KeyError('Wrong distance parameter')

            if method == 'max':
                final_similarity = max(similarities)
            elif method == 'average':
                final_similarity = sum([
                        weights[i] * similarities[i]
                        for i in xrange(len(similarities))
                    ])
            else:
                raise KeyError('Wrong method parameter')

            results.append(is_positive=positive,name=name,score=final_similarity)
        return results

    def pu_learning(self, training=None, testing=None, candidates=None, distance='euclidean', **kwargs):
        if training is None:
            training = self
        if candidates is None:
            candidates = training.spatial_candidates()
        evaluated = candidates
        if testing is not None:
            evaluated += testing.set_pairs
        candidates = self.spatial_candidates()
        random.shuffle(training.set_pairs)
        examples = training.set_pairs[0::2]
        positive = training.set_pairs[1::2]

        name = re.match('.*/([a-z]*).txt',self.filename).group(1)
        timestamp = str(int(time.time()))
        print "SVM timestamp is %s" % timestamp
        train_filename = svm.svm_file_name(name, timestamp, 'train')
        model_filename = svm.svm_file_name(name, timestamp, 'model')
        test_filename = svm.svm_file_name(name, timestamp, 'test')
        prediction_filename = svm.svm_file_name(name, timestamp, 'prediction')

        open(train_filename, "wb").write('\n'.join([rel.svm_transform(examples) for rel in positive + candidates]))
        open(test_filename, "wb").write('\n'.join([rel.svm_transform(examples) for rel in evaluated]))
        os.system('./svm-perf/svm_perf_learn -l 10 -c 0.01 -w 3 %s %s' % (train_filename, model_filename))
        os.system('./svm-perf/svm_perf_classify %s %s %s' % (test_filename, model_filename, prediction_filename))

        svm.svm_timestamp_to_results(timestamp)

    def find_new_pairs(self, n=100, **kwargs):
        if kwargs['method'] == 'max' or kwargs['method'] == 'average':
            results = self.comparative_algorithm(**kwargs)
        if kwargs['method'] == 'pu':
            results = self.pu_learning(**kwargs)
        results.print_top_n_to_file(n, config.output_file)


    def seed_recall(self, size, interesting_relations=None):
        candidates_words = set()
        for rel in self.set_pairs:
            ng_1 = [record[0] for record in model.most_similar(rel.e_1.word, topn=size)]
            ng_2 = [record[0] for record in model.most_similar(rel.e_2.word, topn=size)]
            for word in ["%s-%s" % (e_1, e_2) for e_1 in ng_1 for e_2 in ng_2 if e_1 != e_2]:
                candidates_words.add(word)
        present = 0
        if interesting_relations is None:
            interesting_relations = self.set_pairs
        for rel in interesting_relations:
            present += 1 if rel.word() in candidates_words else 0
        return float(present) / len(interesting_relations)

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

    def get_positions(self, **kwargs):
        positions = []
        for _ in xrange(50):
            testing, training = self.testing_and_training_set(0.2)
            positions.append(self.sim_measure(training, testing, **kwargs))
        return flatten(positions)

for f in glob.glob('./relations/capitals.txt'):
    our_set = PairSet.create_from_file(f)
    our_set.find_new_pairs(method='pu')
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