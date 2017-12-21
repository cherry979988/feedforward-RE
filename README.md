# feedforward-RE
Feedforward NeuralNet for Relation Extraction.

### Data Preparation

Place data in CoType format and brown cluster file obtained in `data/source/$dataname`

### Feature Extraction
``./feature_extraction.sh KBP``

### Train FF-NN model
``python3 FFNN/run.py KBP``

