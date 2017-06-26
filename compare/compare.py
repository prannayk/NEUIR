import json
import sys
import string
printable = string.printable
print(printable)
args = sys.argv[1:]

with open(args[0],mode="r") as f:
    list1 = f.readlines()
    list1 = map(lambda x: filter(lambda y: y != '\n',x).split()[2],list1)[:100]
with open(args[1],mode="r") as f:
    list2 = f.readlines()
    list2 = map(lambda x: filter(lambda y: y != '\n',x).split()[2],list2)[:500]

req_list = []
for tweet in list1:
    if not tweet in list2:
        req_list.append(tweet)

with open("../dataset/%s.jsonl"%(args[2])) as fil:
    t = fil.readlines()

dictionary = dict()
for line in t:
    tweet = json.loads(line)
    dictionary[int(tweet['id'])] = tweet['text']

for tweet in req_list :
    print(filter(lambda x: x in printable, dictionary[int(tweet)]))


