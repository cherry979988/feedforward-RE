import sys

with open(sys.argv[1]) as fin:
  features = {}
  for line in fin:
    seg = line.strip('\r\n').split('\t')
    if int(seg[2])>=2:
      if seg[0] in features:
        print(seg[0], seg[1], features[seg[0]])
      features[seg[0]] = seg[1]
