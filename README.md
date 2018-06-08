# feedforward-RE
Feedforward NeuralNet for Relation Extraction.

### Data Preparation

- Place data in [CoType](https://github.com/shanzhenren/CoType) format obtained in `data/source/$dataname`.
- Run `./brown_clustering.sh KBP` to generate browncluster file (may take a while, don't repeat if nothing in data changed)

### Feature Extraction
First, run the server using all jars in the CoreNLP home directory (in another terminal):
``java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000``
Then:
``./feature_extraction.sh KBP``

### Train FF-NN model
``python3 FFNN/run.py KBP``

