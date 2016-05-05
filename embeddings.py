from libraries import *
from helper import *
from config import *

def file_path(filename):
    return '%s%s' % (svm_folder, filename)


def process_results(results):
    results = sorted(results, key=lambda x: -x[2])
    for i in xrange(len(results)):
        results[i][3] = i+1
    return [float(x[3])/len(results) for x in results if x[0]]


def evaluate_results(results):
    mean = np.mean(results)
    std = np.std(results)
    median = np.median(results)
    return mean, std, median


def svm_file(timestamp, average=True, topn=None):
    test_file = glob.glob("%s*%s-test*" % (svm_folder, timestamp))[0]
    predict_file = glob.glob("%s*%s-prediction*" % (svm_folder, timestamp))[0]
    records = list()
    with open(test_file) as f:
        for line in f:
            positive = (line.split()[0] == '1')
            name = line.split('#')[1].strip()
            records.append([positive, name, 0, 0])
    with open(predict_file) as f:
        i = 0
        for line in f:
            records[i][2] = float(line)
            i += 1
    records = sorted(records, key=lambda x: -x[2])
    records = [res for res in records if res[0] is False]
    for i in xrange(100):
        print "?\t%s" % records[i][1]


def get_timestamp(string):
        return re.match('%s[a-z]*-([0-9]*).*' % svm_folder, string).group(1)


def evaluate_svm_files(name):
    print evaluate_results(flatten([process_results(svm_file(get_timestamp(f))) for f in glob.glob('%s%s*train*' % (svm_folder, name))]))


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

    def testing_slices(self):
        n = 5  # number of slices
        shuffled_relations = list(self.relations)
        random.shuffle(shuffled_relations)
        slices = self.slice_list(shuffled_relations, n)
        parted_slices = [self.part_slices(slices, i) for i in xrange(n)]
        return [(RelationSet(part[0]), RelationSet(part[1])) for part in parted_slices]

    @staticmethod
    def slice_list(l, n):
        n = max(1, n)
        return [l[i::n] for i in xrange(n)]

    @staticmethod
    def part_slices(slices, i):
        testing_set = slices[i]
        training_slices = slices[:i] + slices[i+1:]
        training_set = flatten(training_slices)
        return testing_set, training_set

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

        if weight_type == 'none':
            weights = [float(1)/len(training) for _ in xrange(len(training))]
        elif weight_type == 'normalized':
            weights = normalize_list([training.rel_weight(rel, distance) for rel in training.relations])
        elif weight_type == 'softmax':
            weights = softmax_list([training.rel_weight(rel, distance) for rel in training.relations])
        else:
            raise KeyError('Wrong weight_type parameter')

        results = list()
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

            results.append([positive, name, final_similarity, 0])
        return process_results(results)

    def find_new(self, n=100):
        candidates = self.spatial_candidates()
        results = list()
        #weights = self.softmax_list([self.rel_weight(rel, 'euclidean') for rel in self.relations])
        weights = [float(1)/len(self.relations) for _ in xrange(len(self.relations))]
        for rel in candidates:
            positive = 0
            name = rel.word()
            similarities = [rel.cosine_similarity(x) for x in self.relations]
            final_similarity = sum([
                weights[i] * similarities[i]
                for i in xrange(len(similarities))
            ])
            results.append([positive, name, final_similarity, 0])
        results = sorted(results, key=lambda x: -x[2])
        for result in results[0:n]:
            print result[1]
        self.clear_cache()


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

    def run_sim_test(self):
        print datetime.datetime.now()
        results_1 = []
        results_2 = []
        results_3 = []
        results_4 = []
        results_5 = []
        results_6 = []
        for _ in xrange(20):
            testing, training = self.testing_slices()[0]
            # results_1.append(self.sim_measure(training, testing, distance='euclidean'))
            results_2.append(self.sim_measure(training, testing, distance='euclidean', weight_type='none'))
            # results_3.append(self.sim_measure(training, testing, distance='euclidean', weight_type='normalized'))
            # results_4.append(self.sim_measure(training, testing))
            # results_5.append(self.sim_measure(training, testing, weight_type='none'))
            # results_6.append(self.sim_measure(training, testing, method='max'))
        # print evaluate_results(flatten(results_1))
        print evaluate_results(flatten(results_2))
        # print evaluate_results(flatten(results_3))
        # print evaluate_results(flatten(results_4))
        # print evaluate_results(flatten(results_5))
        # print evaluate_results(flatten(results_6))
        self.clear_cache()


for f in glob.glob('./relations/capitals.txt'):
    print f
    our_set = RelationSet.create_from_file(f)
    candidates = our_set.spatial_candidates()
    relations = our_set.relations
    for i in xrange(20):
        i = i+1
        results = list()
        for j in xrange(100):
            random.shuffle(relations)
            training = RelationSet(relations[0:i])
            testing = RelationSet(relations[-5:])
            res = our_set.sim_measure(training, testing, candidates, distance='euclidean', weight_type='none')
            results.append(res)
        print i, evaluate_results(flatten(results))