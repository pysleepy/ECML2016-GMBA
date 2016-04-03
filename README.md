# ECML2016-GMBA
Experiment codes for GMBA

### Orgainzation:
* **dataset** Data sets applied in the experiment shall be put in this directory.
              Two data sets (emotions and genebase) have been added into this directory as examples.
              Each row of the data set represents an instance. Features and labels shall be saved in two files named
              'dataname_feature.csv' and 'dataname_label.csv' separately. For the label set 'dataname_label.csv', the 
              element marked 1 represents the instance is associated with the label, or 0 on the contrary.
* **CrossValidation** The shuffled data set are kept in this directory for crossvalidation. Files in this directory are
                      generated automaticly by the function LoadingData.generate_validate_data
* **LoadingData.py** Provide basic operations to loading data.
* **BuildGraph.py** Build a graph, which may be used later, of the loaded data set.
* **LaplacianMatrix.py** Get the laplacian matrix and its eigen system of the built graph.
* **SpectralFeatureSelection.py** Spectral feature selection (SPEC).
* **MultiLabelFStatistic.py** Multi label F-Statistic (MLFS).
* **MultiLabelRelief.py** Multi label ReliefF (MLRF).
* **Relief.py** Relief for binary classification problems.
* **FisherScore.py** F-Statistic for binary classification problems.
* **GMBA.py** The graph-margin based algorithm (GMBA).
* **MultiLabelClassification.py** Some basic multi-label classification are implemented in this module.
* **CrossValidationFilter.py** Run cross validation of a multi-label feature selection. (MLFS, SPEC, MLRF, GMBA are allowed)
* **CrossValidationWrapper.py** Run cross validation of a single-label feature selection. It will apply Relief or FisherScore
                                on single-label problems transformed from the multi-label problem
* **Main.py** Interface of the program. There is an example at the bottom of this file. If RAM overflow when running Relief
                                and FisherScore, try Main2 while Main2 is not validated.
