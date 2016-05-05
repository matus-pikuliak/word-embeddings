import math
import random
from helper import *
print_vector_stats([1,2,3,5,9], verbose=False)
exit()

# capitals 1460237289
# cities 1460235363
# currency 1460238812
# family 1460236170
# import math
#
# def score(relevancy, position):
#     return relevancy / math.log(position+1, 2)
#
# def idcg(size):
#     return sum([score(1,pos+1) for pos in xrange(size)])
#
# def dcg(f):
#     i = 1
#     scores = []
#     for line in f:
#         rel = 0
#         if line.startswith('r'):
#             rel = 1
#         if line.startswith('p'):
#             rel = 0.5
#         scores.append(score(rel,i))
#         i+=1
#     return sum(scores)
#
# def ndcg(f):
#     size = len([line for line in f])
#     f.seek(0)
#     return dcg(f) / idcg(size)
#
# dirs = ['pu', 'pu_cos', 'euc', 'cos']
# sets = ['cities', 'currency', 'capitals', 'gender']
# for dir in dirs:
#     for rset in sets:
#         with open('./results/%s/%s.txt' % (dir, rset)) as f:
#             print dir, rset, ndcg(f)
#
# exit()


text = '''1 (0.11317339579908145, 0.19753615420066223, 0.0085866154525129357)
2 (0.10831023361607814, 0.18673354601654679, 0.0054366540086229886)
3 (0.10947644197426683, 0.19357529663951356, 0.0055191010364076375)
4 (0.10489229993258743, 0.18528009022092562, 0.0056257971900113002)
5 (0.093761340103689264, 0.17281944424222428, 0.0043721173851682645)
6 (0.099660706231540344, 0.18032396413842147, 0.0046485574195050273)
7 (0.10170111497480516, 0.19223162307006417, 0.0037949881906757261)
8 (0.11032398772024268, 0.18745584063757478, 0.0051868880126871427)
9 (0.10581947980775294, 0.18303113854408609, 0.0047213047969620692)
10 (0.093014593123917882, 0.17905382914182702, 0.0044836633639357302)
11 (0.10163006503615546, 0.18293987842560641, 0.0041829742037799537)
12 (0.09931462270785138, 0.18301737297308426, 0.0045273117904099554)
13 (0.096917877910501313, 0.17745166273564686, 0.0041466005150514323)
14 (0.086684601320122401, 0.16819314083282197, 0.0025558578613240996)
15 (0.10546290126240951, 0.19276268653068401, 0.0045030626645909417)
16 (0.11147768352950876, 0.19145963415658443, 0.0050753420339196779)
17 (0.1067810837419311, 0.1880733656525973, 0.0046849311082335487)
18 (0.11653580868409694, 0.19488343614009551, 0.0050850416842472823)
19 (0.10671880228717755, 0.18764088579914701, 0.0046267332062679143)
20 (0.086447144180452296, 0.1730213934421887, 0.0024394620573928308)'''
import re
set_size = 206205
for l in text.split('\n'):
    a = re.match('([0-9]*) \((.*), (.*), (.*)\)', l)
    print "%s\t%f\t%f" % (a.group(1), float(a.group(4)) * set_size, float(a.group(3)) * set_size)
exit()


def add(words, word):
    if not words.has_key(word):
        words[word] = 1
    else:
        words[word] += 1

dirs = ['pu', 'pu_cos', 'euc', 'cos']
sets = ['cities', 'currency', 'capitals', 'gender']
for dir in dirs:
    for rset in sets:
        words = dict()
        with open('./results/%s/%s.txt' % (dir, rset)) as f:
            for line in f:
                word_1 = line.split()[1].split('-')[0]
                word_2 = line.split()[1].split('-')[1]
                add(words, word_1)
                add(words, word_2)

        l = [words[rec] for rec in words]
        print dir, rset, float(sum(l)) / len(l)

exit()

dirs = ['pu', 'pu_cos', 'euc', 'cos']
sets = ['cities', 'currency', 'capitals', 'gender']
for dir_1 in dirs:
    for dir_2 in dirs:
        print dir_1, dir_2
        for rset in sets:
            s_1 = set()
            with open('./results/%s/%s.txt' % (dir_1, rset)) as f:
               for line in f:
                   s_1.add(line.split('\t')[1])
            i = 0
            with open('./results/%s/%s.txt' % (dir_2, rset)) as f:
               for line in f:
                   if line.split('\t')[1] in s_1:
                       i+=1
            print rset, i


exit()

dirs = ['pu', 'euc', 'cos', 'pu_cos']
sets = ['cities','currency','capitals','gender']
for dir in dirs:
    for set in sets:
        with open('./results/%s/%s.txt' % (dir,set)) as f:
            a = [0,0,0]
            for line in f:
                if line.startswith('r'):
                    a[0] += 1
                if line.startswith('p'):
                    a[1] += 1
                if line.startswith('w'):
                    a[2] += 1
            print dir, set, a

exit()

names = ['cities','currency','capitals','gender']
n = -1
results = list([[0,0,0,0] for _ in xrange(101)])
for name in names:
    n+=1
    with open('./results/euc/%s.txt' % name) as f:
        a = 0
        i = 0
        for line in f:
            i += 1
            if line.startswith('r') or line.startswith('p'):
                a += 1
            results[i][n] = a
i = 0
for res in results:
    print "%i\t%i\t%i\t%i\t%i" % (i, res[0], res[1], res[2], res[3])
    i+=1

def fcia(results, i):
    return [res[i] for res in results]



#otvor subor
#postupne prechadzaj riadku a ukladaj do polia
#ak riadok zacina r alebo p, predchadzajuce pole + 1
#inak to iste
#vypis pol

exit()

a = 162684+170948+167811+169864+168198+161778+167711+168150+163533+165776+167996+167507+169943+168514+170211+171954+166869+165296+168452+171059
print float(a) / 20 * 3.50
exit()
import glob
for fn in glob.glob('/media/piko/Decko/bcp/dp-data/python-files/new_predictions/*1461432193*prediction*'):
    buckets = [0 for _ in xrange(100)]
    with open(fn) as f:
        for line in f:
            num = float(line)
            bucket_num = int((num - 0.1) / 0.3 * 100)
            buckets[bucket_num] += 1
    for i in xrange(len(buckets)):
        value = float(i) / 100 * 0.3 + 0.1 + 0.0015
        print "%f\t%f" % (value, float(buckets[i])/sum(buckets))
exit()
values = list()
for fn in glob.glob('/media/piko/Decko/bcp/dp-data/python-files/new_predictions/*prediction*'):
    print fn
    with open(fn) as f:
        l = sorted([float(line) for line in f], key=lambda x: -x)
        print "1\t%f" % l[0]
        act = l[0]
        for i in xrange(len(l)):
            if l[i] < act - 0.001:
                act = l[i]
                print "%i\t%f" % (i+1, l[i])
    print
exit()
