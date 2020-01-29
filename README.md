# Association analysis of naturalistic stimuli

A project for the analysis of spatiotemporal co-occurrences in a large, naturalistic movie dataset.

### Data collection

./data-collection/tag_movie.py - send movies which are located in an AWS S3 bucket to be automatically tagged using Amazon Rekognition.

./data-collection/create_itemsets.py - retrieve the raw output files from Rekognition (located in ./data) and process them into an array with each row containing item strings which occurred in the same 200 ms interval. 

### Analysis

regression.py - perform Ridge Regression for the top most frequent items in the dataset of 40,000,000 labels. Retrieve the coefficient matrices.

plotting.py - plot the coefficient and p value matrices.
plot_timecourse.py - plot the timecourses of the frequent items.

### Results
Figures for various analyses are stored in ./results/ridge_regression/figs
The analysis was done for increasing time intervals
It was found that there was a strong correlative structure over an extended period of time, with the trend beginning to decrease at approx. 15 mins.

Figures have been ordered either according to a semantic measure of similarity between object (Leacock-Chodrow LCH measure)
OR
By the order which emerges from hierarchical clustering (as derived in ./results/ridge_regression/investigateCluster.py)


...
