from gensim.models import Word2Vec
from scipy.spatial import distance

top100k = set()
with open('/media/piko/DATA/dp-data/wiki-100k.txt') as f:
    for line in f:
        if not line.startswith('#'):
            top100k.add(line.strip())

model = None
model = Word2Vec.load_word2vec_format('/media/piko/DATA/dp-data/GoogleNews-vectors-negative300.bin', binary=True, selected_words=top100k)
embeddings = dict()


def flatten(l):
    return [item for sublist in l for item in sublist]


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
        name = "%s - %s" % (self.word, embedding.word)
        return emb(vector=vector, word=name)

    def __add__(self, embedding):
        if len(self) != len(embedding):
            raise KeyError('The embeddings have different lengths')
        vector = [self.v[i] + embedding.v[i] for i in xrange(len(self))]
        name = "%s + %s" % (self.word, embedding.word)
        return emb(vector=vector, word=name)

    def neighbours(self, n=100):
        return [emb(record[0]) for record in model.most_similar(self.word, topn=n)]


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
        ng_1 = self.e_1.neighbours
        ng_2 = self.e_2.neighbours
        return [Relation(e_1, e_2, candidate=True) for e_1 in ng_1 for e_2 in ng_2]


class RelationSet:
    def __init__(self, relations, filename=None):
        self.relations = relations
        self.filename = filename

    def __len__(self):
        return len(self.relations)

    def testing_slices(self):
        n = 5 # number of slices
        slices = self.slice_list(self.relations, n)
        parted_slices = [self.part_slices(slices, i) for i in xrange(n)]
        return [(RelationSet(part[0]), RelationSet(part[1])) for part in parted_slices]

    def spatial_candidates(self):
        return flatten([rel.candidates for rel in self.relations])

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

for training, testing in our_set.testing_slices():
    positives = training.relations
    candidates = training.spatial_candidates()
    model = pu_learning(positives, candidates)
    results = predict(testing.relations, candidates)
    evaluate(results)

#1. PU learning
#Rozdel set na dve casti 80-20 5x
#Pre kazde rozdelenie
    #Urob triedu kandidatov
    #Nakrm to cele do klasifikatora
    #Vyhodnot na kandidatoch + testing

# Urob to iste, ale namiesto vektorov daj podobnosti

#2. Ukazkovy vektor
#Vytvor ukazkovy vektor z triedy ako:
#   a) najlepsi jedinec
#   b) priemer
#   c) vazeny priemer
#Tento vektor aplikuj na vsetky slova v triede
#Vyhodnot novonajdene dvojice s testovacimi




