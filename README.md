# feedforward-RE
Feedforward NeuralNet for Relation Extraction.

### Data Preparation

- Place data in CoType format obtained in `data/source/$dataname`.
- Run `python3 generateBClusterInput.py` in respective data source folder.
- `./brown_clustering.sh KBP` to generate browncluster file (may take a while, don't repeat if nothing in data changed)

### Feature Extraction
``./feature_extraction.sh KBP``

### Train FF-NN model
``python3 FFNN/run.py KBP``

