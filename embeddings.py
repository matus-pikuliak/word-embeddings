from libraries import *
import results_helper as res
import config
from helper import *

def file_path(filename):
    return '%s%s' % (config.svm_folder, filename)


def svm_file(timestamp):
    test_file = glob.glob("%s*%s-test*" % (config.svm_folder, timestamp))[0]
    predict_file = glob.glob("%s*%s-prediction*" % (config.svm_folder, timestamp))[0]
    results = res.ResultList()
    with open(test_file) as f:
        for line in f:
            positive = (line.split()[0] == '1')
            name = line.split('#')[1].strip()
            results.append(is_positive=positive, name=name)
    with open(predict_file) as f:
        i = 0
        for line in f:
            results[i].ndcg_score = float(line)
            i += 1
    return results


def get_timestamp(string):
    return re.match('%s[a-z]*-([0-9]*).*' % config.svm_folder, string).group(1)

def evaluate_svm_files(name):
    files = glob.glob('%s%s*prediction*' % (config.svm_folder, name))
    timestamps = [get_timestamp(filename) for filename in files]
    positions = flatten([svm_file(timestamp).positive_positions() for timestamp in timestamps])
    print_vector_stats(positions)

evaluate_svm_files('capitals')
exit()




# for f in glob.glob('./relations/*.txt'):
#     name = re.match('.*/([a-z]*).txt',f).group(1)
#     print name
#     evaluate_svm_files(name)


def svm_transform(l):
    return ' '.join(["%d:%f" % (i+1, l[i]) for i in xrange(len(l))])

top100k = set()
with open('./wiki-100k.txt') as f:
    for line in f:
        if not line.startswith('#'):
            top100k.add(line.strip())
model = Word2Vec.load_word2vec_format('/media/piko/Decko/bcp/dp-data/GoogleNews-vectors-negative300.bin', binary=True, selected_words=top100k)

embeddings = dict()
def emb(word=None, vector=None):
    if word not in embeddings:
        embeddings[word] = Embedding(word=word, vector=vector)
    return embeddings[word]


cosines = dict()
def cosine_similarity(emb_1, emb_2):
    if emb_2 in cosines and emb_1 in cosines[emb_2]:
        return cosines[emb_2][emb_1]
    if emb_1 in cosines and emb_2 in cosines[emb_1]:
        return cosines[emb_1][emb_2]
    if emb_1 not in cosines:
        cosines[emb_1] = dict()
    cosine_value = -distance.cosine(emb_1.v, emb_2.v) / 2 + 1
    cosines[emb_1][emb_2] = cosine_value
    return cosine_value

euclideans = dict()
def euclidean_similarity(emb_1, emb_2):
    if emb_2 in euclideans and emb_1 in euclideans[emb_2]:
        return euclideans[emb_2][emb_1]
    if emb_1 in euclideans and emb_2 in euclideans[emb_1]:
        return euclideans[emb_1][emb_2]
    if emb_1 not in euclideans:
        euclideans[emb_1] = dict()
    euclidean_value = 1 / (1 + distance.euclidean(emb_1.v, emb_2.v))
    euclideans[emb_1][emb_2] = euclidean_value
    return euclidean_value


class Embedding:

    def __init__(self, word=None, vector=None):
        if vector is None and word is None:
            raise KeyError('You have to state word or vector')
        self.v = model[word] if vector is None else vector
        self.word = word

    def __len__(self):
        return len(self.v)

    def __sub__(self, embedding):
        if len(self) != len(embedding):
            raise KeyError('The embeddings have different lengths')
        vector = [self.v[i] - embedding.v[i] for i in xrange(len(self))]
        name = "%s-%s" % (embedding.word, self.word)
        return emb(vector=vector, word=name)

    def __add__(self, embedding):
        if len(self) != len(embedding):
            raise KeyError('The embeddings have different lengths')
        vector = [self.v[i] + embedding.v[i] for i in xrange(len(self))]
        name = "%s+%s" % (embedding.word, self.word)
        return emb(vector=vector, word=name)

    def neighbours(self, n=100):
        return [emb(record[0]) for record in model.most_similar(self.word, topn=n)]

    def svm_sim_transform(self, relations):
        return svm_transform([self.euclidean_similarity(rel.rel_embedding) for rel in relations])


class Relation:
    def __init__(self, embedding_1, embedding_2, positive=False, candidate=False):
        self.e_1 = embedding_1
        self.e_2 = embedding_2
        self.rel_embedding = embedding_2 - embedding_1
        self.positive = positive
        self.candidate = candidate

    def __len__(self):
        return len(self.rel_embedding)

    def word(self):
        return self.rel_embedding.word

    def cosine_similarity(self, relation):
        return cosine_similarity(self.rel_embedding, relation.rel_embedding)

    def euclidean_similarity(self, relation):
        return euclidean_similarity(self.rel_embedding, relation.rel_embedding)

    def spatial_candidates(self):
        ng_1 = self.e_1.neighbours()
        ng_2 = self.e_2.neighbours()
        return [Relation(e_1, e_2, candidate=True) for e_1 in ng_1 for e_2 in ng_2 if e_1 != e_2]

    def svm_standard_transform(self):
        return '%d %s # %s' % (self.svm_target(), svm_transform(self.rel_embedding.v), self.rel_embedding.word)

    def svm_sim_transform(self, relations):
        return '%d %s # %s' % (self.svm_target(), self.rel_embedding.svm_sim_transform(relations), self.rel_embedding.word)

    def svm_target(self):
        if self.positive:
            return 1
        if self.candidate:
            return -1
        raise KeyError('Relation should be marked candidate or positive.')


class RelationSet:
    def __init__(self, relations, filename=None):
        self.relations = relations
        self.filename = filename

    def __len__(self):
        return len(self.relations)

    def spatial_candidates(self, filter_positives=True):
        preliminary_candidates = flatten([rel.spatial_candidates() for rel in self.relations])
        keys = set([rel.word() for rel in self.relations]) if filter_positives else set()
        candidates = list()
        for candidate in preliminary_candidates:
            if candidate.word() not in keys:
                keys.add(candidate.word())
                candidates.append(candidate)
        return candidates

    def rel_weight(self, relation, distance):
        if distance == 'euclidean':
            similarities = [relation.euclidean_similarity(rel) for rel in self.relations]
        elif distance == 'cosine':
            similarities = [relation.cosine_similarity(rel) for rel in self.relations]
        else:
            raise KeyError('Wrong distance parameter')
        return (sum(similarities) - 1) / (len(similarities) - 1)

    def sim_measure(self, training, testing, candidates=None, method='average', weight_type='softmax', distance='cosine'):

        if candidates is None:
            candidates = training.spatial_candidates()

        print len(candidates)

        if weight_type == 'none':
            weights = [float(1)/len(training) for _ in xrange(len(training))]
        elif weight_type == 'normalized':
            weights = normalize_list([training.rel_weight(rel, distance) for rel in training.relations])
        elif weight_type == 'softmax':
            weights = softmax_list([training.rel_weight(rel, distance) for rel in training.relations])
        else:
            raise KeyError('Wrong weight_type parameter')

        results = res.ResultList()
        evaluated = testing.relations + candidates
        for rel in evaluated:
            positive = rel.positive
            name = rel.word()

            if distance == 'cosine':
                similarities = [rel.cosine_similarity(x) for x in training.relations]
            elif distance == 'euclidean':
                similarities = [rel.euclidean_similarity(x) for x in training.relations]
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
        return results.positive_positions()

    def find_new(self, n=100):
        candidates = self.spatial_candidates()
        results = res.ResultList()
        #weights = self.softmax_list([self.rel_weight(rel, 'euclidean') for rel in self.relations])
        weights = [float(1)/len(self.relations) for _ in xrange(len(self.relations))]
        for rel in candidates:
            name = rel.word()
            similarities = [rel.cosine_similarity(x) for x in self.relations]
            final_similarity = sum([
                weights[i] * similarities[i]
                for i in xrange(len(similarities))
            ])
            results.append(is_positive=False, name=name, score=final_similarity)
        results.print_top_n(100)


    @classmethod
    def create_from_file(cls, filename, capitalize=False):
        def create_relation(pair):
            return Relation(emb(pair[0]), emb(pair[1]), positive=True)

        def capitalize_line(pair):
            return map(str.capitalize, pair)

        with open(filename) as f:
            pairs = [line.strip().split() for line in f]
            if capitalize:
                pairs = map(capitalize_line, pairs)
            relations = map(create_relation, pairs)
            return RelationSet(relations, filename=filename)

    def seed_recall(self, size, interesting_relations=None):
        candidates_words = set()
        for rel in self.relations:
            ng_1 = [record[0] for record in model.most_similar(rel.e_1.word, topn=size)]
            ng_2 = [record[0] for record in model.most_similar(rel.e_2.word, topn=size)]
            for word in ["%s-%s" % (e_1, e_2) for e_1 in ng_1 for e_2 in ng_2 if e_1 != e_2]:
                candidates_words.add(word)
        present = 0
        if interesting_relations is None:
            interesting_relations = self.relations
        for rel in interesting_relations:
            present += 1 if rel.word() in candidates_words else 0
        return float(present) / len(interesting_relations)

    def spatial_candidates_size(self, size):
        candidates_words = set()
        for rel in self.relations:
            ng_1 = [record[0] for record in model.most_similar(rel.e_1.word, topn=size)]
            ng_2 = [record[0] for record in model.most_similar(rel.e_2.word, topn=size)]
            for word in ["%s-%s" % (e_1, e_2) for e_1 in ng_1 for e_2 in ng_2 if e_1 != e_2]:
                candidates_words.add(word)
        return len(candidates_words)

    @staticmethod
    def clear_cache():
        cosines.clear()
        euclideans.clear()
        embeddings.clear()

    def find_new_svm(self):
        name = re.match('.*/([a-z]*).txt',self.filename).group(1)
        candidates = self.spatial_candidates()
        examples = self.relations[0::2]
        training = self.relations[1::2]
        timestamp = int(time.time())
        train_filename = file_path('%s-%d-train-sim_svm' % (name,timestamp))
        model_filename = file_path('%s-%d-model-sim_svm' % (name,timestamp))
        test_filename = file_path('%s-%d-test-sim_svm' % (name,timestamp))
        prediction_filename = file_path('%s-%d-prediction-sim_svm' % (name,timestamp))
        open(train_filename, "wb").write('\n'.join([rel.svm_sim_transform(examples) for rel in training + candidates]))
        open(test_filename, "wb").write('\n'.join([rel.svm_sim_transform(examples) for rel in candidates]))
        os.system('./svm-perf/svm_perf_learn -l 10 -c 0.01 -w 3 %s %s' % (train_filename, model_filename))
        os.system('./svm-perf/svm_perf_classify %s %s %s' % (test_filename, model_filename, prediction_filename))

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
        shuffled = list(self.relations)
        random.shuffle(shuffled)
        return RelationSet(shuffled[0:size]), RelationSet(shuffled[size:])

    def get_positions(self, **kwargs):
        positions = []
        for _ in xrange(50):
            testing, training = self.testing_and_training_set(0.2)
            positions.append(self.sim_measure(training, testing, **kwargs))
        return flatten(positions)


for f in glob.glob('./relations/capitals.txt'):
    our_set = RelationSet.create_from_file(f)
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