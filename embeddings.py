from gensim.models import Word2Vec
from scipy.spatial import distance
import os, time, random, re, glob

data_folder = '/media/piko/DATA/dp-data/python-files/'


def file_path(filename):
    return '%s%s' % (data_folder, filename)

# def evaluate_svm_files(timestamp, average=True, topn=None):
#     test_file = glob.glob("%s%s-test*" % (data_folder, timestamp))[0]
#     predict_file = glob.glob("%s%s-prediction*" % (data_folder, timestamp))[0]
#     records = list()
#     with open(test_file) as f:
#         for line in f:
#             positive = (line.split()[0] == '1')
#             name = line.split('#')[1].strip()
#             records.append([positive, name, 0, 0])
#     with open(predict_file) as f:
#         i = 0
#         for line in f:
#             records[i][2] = float(line)
#             i += 1
#     records = sorted(records, key=lambda x: -x[2])
#     for i in xrange(len(records)):
#         records[i][3] = i+1
#     print [x[3] for x in records if x[0]]
#
# for f in glob.glob('%s*' % data_folder):
#     match = re.match('.*files/([0-9]*)-train.*', f)
#     if match is not None:
#         evaluate_svm_files(match.group(1))

def svm_transform(l):
    return ' '.join(["%d:%f" % (i+1, l[i]) for i in xrange(len(l))])

top100k = set()
with open('./wiki-100k.txt') as f:
    for line in f:
        if not line.startswith('#'):
            top100k.add(line.strip())
model = Word2Vec.load_word2vec_format('/media/piko/DATA/dp-data/GoogleNews-vectors-negative300.bin', binary=True, selected_words=top100k)

def flatten(l):
    return [item for sublist in l for item in sublist]

embeddings = dict()
def emb(word=None, vector=None):
    if word not in embeddings:
        embeddings[word] = Embedding(word=word, vector=vector)
    return embeddings[word]


class Embedding:
    def __init__(self, word=None, vector=None):
        if vector is None and word is None:
            raise KeyError('You have to state word or vector')
        self.v = model[word] if vector is None else vector
        self.word = word

    def cosine_similarity(self, embedding):
        return -distance.cosine(self.v, embedding.v) / 2 + 1

    def euclidean_similarity(self, embedding):
        return 1 / (1 + distance.euclidean(self.v, embedding.v))

    def __len__(self):
        return len(self.v)

    def __sub__(self, embedding):
        if len(self) != len(embedding):
            raise KeyError('The embeddings have different lengths')
        vector = [self.v[i] - embedding.v[i] for i in xrange(len(self))]
        name = "%s-%s" % (self.word, embedding.word)
        return emb(vector=vector, word=name)

    def __add__(self, embedding):
        if len(self) != len(embedding):
            raise KeyError('The embeddings have different lengths')
        vector = [self.v[i] + embedding.v[i] for i in xrange(len(self))]
        name = "%s+%s" % (self.word, embedding.word)
        return emb(vector=vector, word=name)

    def neighbours(self, n=100):
        return [emb(record[0]) for record in model.most_similar(self.word, topn=n)]

    def svm_sim_transform(self, relations):
        return svm_transform([self.cosine_similarity(rel.rel_embedding) for rel in relations])


class Relation:
    def __init__(self, embedding_1, embedding_2, positive=False, candidate=False):
        self.e_1 = embedding_1
        self.e_2 = embedding_2
        self.rel_embedding = embedding_2 - embedding_1
        self.positive = positive
        self.candidate = candidate

    def __len__(self):
        return len(self.rel_embedding)

    def cosine_similarity(self, relation):
        return self.rel_embedding.cosine_similarity(relation.rel_embedding)

    def euclidean_similarity(self, relation):
        return self.rel_embedding.euclidean_similarity(relation.rel_embedding)

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

    def spatial_candidates(self):
        return flatten([rel.spatial_candidates() for rel in self.relations])

    def naive_svm_generate_files(self):
        for testing, training in self.testing_slices():
            candidates = training.spatial_candidates()
            timestamp = int(time.time())
            train_filename = file_path('%d-train-naive_svm' % timestamp)
            model_filename = file_path('%d-model-naive_svm' % timestamp)
            test_filename = file_path('%d-test-naive_svm' % timestamp)
            prediction_filename = file_path('%d-prediction-naive_svm' % timestamp)
            open(train_filename, "wb").write('\n'.join([rel.svm_standard_transform() for rel in training.relations + candidates]))
            open(test_filename, "wb").write('\n'.join([rel.svm_standard_transform() for rel in testing.relations + candidates]))
            os.system('./svm-perf/svm_perf_learn -l 10 -c 0.01 -w 3 %s %s' % (train_filename, model_filename))
            os.system('./svm-perf/svm_perf_classify %s %s %s' % (test_filename, model_filename, prediction_filename))

    def similarity_svm_generate_files(self):
        for testing, training in self.testing_slices():
            candidates = training.spatial_candidates()
            examples = self.relations[0::2]
            training = self.relations[1::2]
            timestamp = int(time.time())
            train_filename = file_path('%d-train-sim_svm' % timestamp)
            model_filename = file_path('%d-model-sim_svm' % timestamp)
            test_filename = file_path('%d-test-sim_svm' % timestamp)
            prediction_filename = file_path('%d-prediction-sim_svm' % timestamp)
            open(train_filename, "wb").write('\n'.join([rel.svm_sim_transform(examples) for rel in training + candidates]))
            open(test_filename, "wb").write('\n'.join([rel.svm_sim_transform(examples) for rel in testing.relations + candidates]))
            os.system('./svm-perf/svm_perf_learn -l 10 -c 0.01 -w 3 %s %s' % (train_filename, model_filename))
            os.system('./svm-perf/svm_perf_classify %s %s %s' % (test_filename, model_filename, prediction_filename))

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

    @staticmethod
    def slice_list(l, n):
        n = max(1, n)
        return [l[i:i + n] for i in range(0, len(l), n)]

    @staticmethod
    def part_slices(slices, i):
        testing_set = slices[i]
        training_slices = slices[:i] + slices[i+1:]
        training_set = flatten(training_slices)
        return testing_set, training_set


our_set = RelationSet.create_from_file('/home/piko/RubymineProjects/dp/relations/mikolov/11a-capital-common-countries.txt', capitalize=True)
our_set.similarity_svm_generate_files()



#2. Ukazkovy vektor
#Vytvor ukazkovy vektor z triedy ako:
#   a) najlepsi jedinec
#   b) priemer
#   c) vazeny priemer
#Tento vektor aplikuj na vsetky slova v triede
#Vyhodnot novonajdene dvojice s testovacimi




