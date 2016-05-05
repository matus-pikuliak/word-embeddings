from libraries import *

"""
These dictionaries are used as for caching results in our system.
embeddings - Contains Embedding objects cached by the name (e.g. 'Paris', but also pair embedding such as 'Paris-France')
cosines    - Contains calculated cosine similarities between Embedding objects
euclideans - Contains calculated euclidean similarities between Embedding objects
"""
embeddings = dict()
cosines = dict()
euclideans = dict()


def cached_embedding(word):
    """
    Interface to 'embeddings' dictionary
    :param word: string
    :return: Embedding or None
    """
    if word in embeddings:
        return embeddings[word]
    return None


def cosine_similarity(emb_1, emb_2):
    """
    Calculates cosine similarity between emb_1 and emb_2 Embedding objects. It will store the calculations
    and if the same embeddings are compared again it will return this saved result.
    :param emb_1: Embedding
    :param emb_2: Embedding
    :return: float
    """
    if emb_2 in cosines and emb_1 in cosines[emb_2]:
        return cosines[emb_2][emb_1]
    if emb_1 in cosines and emb_2 in cosines[emb_1]:
        return cosines[emb_1][emb_2]
    if emb_1 not in cosines:
        cosines[emb_1] = dict()
    cosine_value = -distance.cosine(emb_1.v, emb_2.v) / 2 + 1
    cosines[emb_1][emb_2] = cosine_value
    return cosine_value


def euclidean_similarity(emb_1, emb_2):
    """
    Calculates euclidean similarity between emb_1 and emb_2 Embedding objects. It will store the calculations
    and if the same embeddings are compared again it will return this saved result.
    :param emb_1: Embedding
    :param emb_2: Embedding
    :return: float
    """
    if emb_2 in euclideans and emb_1 in euclideans[emb_2]:
        return euclideans[emb_2][emb_1]
    if emb_1 in euclideans and emb_2 in euclideans[emb_1]:
        return euclideans[emb_1][emb_2]
    if emb_1 not in euclideans:
        euclideans[emb_1] = dict()
    euclidean_value = 1 / (1 + distance.euclidean(emb_1.v, emb_2.v))
    euclideans[emb_1][emb_2] = euclidean_value
    return euclidean_value


def clear_cache():
    """
    Calculates cosine similarity between emb_1 and emb_2 Embedding objects. It will store the calculations
    and if the same embeddings are compared again it will return this saved result.
    :param emb_1: Embedding
    :param emb_2: Embedding
    :return: float
    """
    global embeddings, cosines, euclideans
    embeddings = dict()
    cosines = dict()
    euclideans = dict()
