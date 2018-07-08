There are 3 main files that can be run using the command line: "main_network.py", "main_ablation.py" and "main_baseline_model.py".
	- "main_network.py" is for training the network, displaying the results (accuracy, auc, confusion matrices) of the network, and output the created dataframes to csv files
	- "main_ablation.py" is for carrying out ablation study and displaying the results (auc scores etc.) of the network with each ablation method
	- "main_baseline_model.py" is for creating and displaying the result (auc scores etc.) of baseline logistic regression model with handcrafted features

The classes definitions are in source files found in sub directories "load_data", "network", "create_dataframe", "ablation", "baseline_model". The classes offer more detailed information e.g. auc scores of each of the 5 folds.

Helper functions used in classes are in the sub directory "utils".

If there are any questions on how these codes work, contact me at tue.pham@mail.utoronto.ca or minhtue96@gmail.com