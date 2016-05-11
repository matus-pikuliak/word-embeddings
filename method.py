import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input",
                    help="filename of seed set file")
parser.add_argument("-o", dest='output',
                    help="filename where results will be written")
parser.add_argument("-t", dest='result_count',
                    type=int, default=100,
                    help="number of results returned")
parser.add_argument("-m", dest='method',
                    type=int, choices=[1, 2, 3], default=1,
                    help="rating method (1 - avg, 2 - max, 3 - pu)")
parser.add_argument("-n", dest='neighborhood',
                    type=int, default=100,
                    help="number of neighbours selected when generating candidates")
parser.add_argument("-d", dest='similarity',
                    type=int, choices=[1, 2], default=1,
                    help="similarity measure (1 - euclidean, 2 - cosine)")
parser.add_argument("-s", dest='normalization',
                    type=int, choices=[1, 2, 3], default=1,
                    help="normalization method applied to avg method (1 - none, 2 - standard, 3 - softmax)")
args = parser.parse_args()


def method_names(x):
    return {
        1: 'avg',
        2: 'max',
        3: 'pu'
    }[x]


def similarity_names(x):
    return {
        1: 'euclidean',
        2: 'cosine',
    }[x]


def normalization_names(x):
    return {
        1: 'none',
        2: 'standard',
        3: 'softmax'
    }[x]

if not os.path.isfile(args.input):
    raise KeyError('Given file does not exist')
import embeddings as emb
seed_set = emb.PairSet.create_from_file(args.input)
seed_set.find_new_pairs(output=args.output,
                        result_count=args.result_count,
                        method=method_names(args.method),
                        neighborhood=args.neighborhood,
                        distance=similarity_names(args.similarity),
                        normalization=normalization_names(args.normalization))



