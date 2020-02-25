# Association analysis of naturalistic stimuli

A project for the analysis of temporal co-occurrences in a large, naturalistic movie dataset. This work was completed in Trinity College Dublin under the supervision of Prof. Rhodri Cusack.

## Motivation
Many large and curated datasets are unnaturalistic and contain specifically designed, esoteric subclasses. In order to better understand human cognition using models of vision and learning (e.g. deep neural networks) more naturalistic datasets are required.

In this work, we were interested in generating a visual dataset with preserved *temporal* structure. We chose 100 live action movies which contained typical scenarios and visual scenes that one may encounter throughout life. These movies were tagged using Amazon Web Services' Rekognition, and subsquent analyses examined the correlative structure of associated items across the entire dataset.

This analysis probed associations at a higher level than is typically examined in vision reasearch. Instead of testing the feature level clustering and association that occurs in object recognition, we aimed to investigate how objects relate to other objects in their surrounding context, and whether this signal can be learned to form more semantically meaningful category clusters.

### Data collection

./data-collection/tag_movie.py - this script sends movies which are located in an AWS S3 bucket to be automatically tagged using Amazon Rekognition. The program will return a pickle file to the specified s3 bucket with all labels and data for the requested video.

./data-collection/create_itemsets.py - retrieve the raw output files from Rekognition (located in ./data) and process them into an array with each row containing item strings which occurred in the same 200 ms interval. 

**Summary of the dataset**


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
