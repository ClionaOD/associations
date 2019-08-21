# associations

The working repository for association analysis of movies using Amazon Rekognition.

## tag_movies, s3tools and videotools
Scripts for sending videos stored in s3 bucket to Rekognition and processing the results through SQS and SNS.
Labels are saved to a .pickle file in a new s3 bucket.

## movie_analysis_v2
Script for analysing the .pickle movie files (must first be downloaded to working directory).

Labels are:
   1) Processed into 'baskets' (lists of labels) containing object labels.
   2) Pooled across baskets from the 200 ms frames according to a desired length.
   3) Passed through an association rule-mining apriori algorithm to find frequently occuring itemsets.
   4) Frequent itemsets are processed for association rules of various metrics including support, confidence, lift and leverage.
   5) An optional shuffling of the baskets can be performed in order to test for noise in the distribution of the association metrics.

## Results
### plot_histogram
Simple script for examining the distribution of leverage in the data.

### plot_matrix_v2
Script for plotting leverage of all itemsets in a symmetrical matrix using Pandas.
The matrix is then visualised as an RDM and hierarchical clustering performed to examine sensible groupings in the co-occurrence data.
