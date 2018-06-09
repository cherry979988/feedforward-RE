__author__ = 'ZeqiuWu'
import json
import sys
from collections import defaultdict
import unicodedata


file = open('./train.json', 'r')
f = open('./bc_input.txt', 'w')
writtenSents = set()
for line in file.readlines():
    sent = json.loads(line)
    sentText = sent['sentText'].rstrip('\n').rstrip('\r')
    if sentText in writtenSents:
        continue
    f.write(sentText)
    f.write('\n')
    writtenSents.add(sentText)
file.close()

file = open('./test.json', 'r')
for line in file.readlines():
    sent = json.loads(line)
    sentText = sent['sentText'].rstrip('\n').rstrip('\r')
    if sentText in writtenSents:
        continue
    f.write(sentText)
    f.write('\n')
    writtenSents.add(sentText)
file.close()
f.close()
