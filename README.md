## Description of Files:
1. EDA.ipynb - Jupyter notebook for EDA of the training data.
2. Signature_FE_and_Model.ipynb - Jupyter notebook for loading the training data, training model pipeline and testing the F1 score over the test.
3. Model analysis.ipynb - Jupyter notebook for testing different aspects and metrics over our models.
4. model_hyperparameter_tuning.py - Script for performing grid search for a given model pipeline.
5. tune_non_significant_cols.py - Script for finding the non statistically significant columns to be removed before training a model.
6. tune_signature_model.py - Script for tuning signature's hyper parameters.
7. utils/data_handler.py - Script containing several functions for data loading and processing, main function is: get_model_prepared_dataset, which loads dataset from a given folder.
8. utils/feature_selection.py - Script containing functions for selecting features prior to model training and prediction.
9. utils/signature.py  - Script for creating signature per patient prior to model training and prediction.
10. utils/statistics.py - Script for statistical tests and analysis to use in EDA notebook.
