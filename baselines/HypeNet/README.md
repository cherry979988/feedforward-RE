# HypeNet
Improving Hypernymy Detection with an Integrated Path-based and Distributional Method, Vered Shwartz, Yoav Goldberg, Ido Dagan. (https://arxiv.org/pdf/1603.06076)

### Data Preparation
- KBP, NYT dataset: download ``train.txt``, ``test.txt`` and ``relation2id.txt`` from https://drive.google.com/file/d/1Xn3tA89wfePlh2OgHU7cw3Lh5RkjIclW/view (link provided by Frank Xu), place files in ``data/KBP/`` or ``data/NYT/``

- TACRED dataset: download ``train_new.json`` and ``test_new.json`` to ``data/TACRED/`` from the above link, then ``python3 preprocess.py`` 
(this preprocess code combines the one in TensorFlow-NRE and the one in HypeNet. Input '_new.json' files and output '.txt' files applicable to HypeNet)

- Pre-trained GloVe representation: download ``glove.6B.100d.txt`` from http://nlp.stanford.edu/data/glove.6B.zip and place it in ``data/``

### Obtain Shortest Dependency Path
- First, run the server using all jars in the CoreNLP home directory (in another terminal):
``java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000``
- Then:
``python3 shortest_dep.py``
(Dataset name is hardcoded; also, need to run twice with train and test separately)

### Train HypeNet Model
- ``python3 sdp.py``
(Dataset name is hardcoded)


### Requirements
- keras 2.0
