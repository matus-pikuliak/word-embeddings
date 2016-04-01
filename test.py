import plyvel
import struct
import sys
import datetime
from lshash import LSHash


#filename = sys.argv[1]
#vector_arguments = sys.argv[2:]
#vector_model = [float(scalar) for scalar in vector_arguments]

db = plyvel.DB('/home/piko/RubymineProjects/dp/leveldb/deps-levy_copy')
words = [word for word, _ in db]
lsh = LSHash(10, 300, 20)
query_vectors = {}
vector_bytes = db.get('play')
vector_model = list(struct.unpack('300f', vector_bytes))

for word in words:
    vector_bytes = db.get(word)
    word_vector = list(struct.unpack('300f', vector_bytes))
    lsh.index(word_vector, word)

    #for i in range(len(vector_model)):
        #word_vector[i] += vector_model[i]
    query_vectors[word] = word_vector

result = lsh.query(query_vectors['play'], 5, 'cosine')
print result[0][0][1]
print result[1][0][1]
print result[2][0][1]
print result[3][0][1]
print result[4][0][1]

exit()

#output_file = open(filename, 'w')
#output_string = ""
timeA = datetime.datetime.now()
i = 0
for word in words:
    #new_line = word
    if i%1000 == 0:
        print i
        timeB = datetime.datetime.now()
        print (timeB - timeA).seconds
    i += 1
    result = lsh.query(query_vectors[word], 5, 'cosine')
    #for row in result:
        #new_line += " "
        #new_line += row[0][1]
    #output_string += new_line + '\n'
    #if len(output_string) > 20000:
        #output_file.write(output_string)
        #output_string = ""