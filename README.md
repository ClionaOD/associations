# Analysing the temporal associations of objects in a naturalistic dataset

A project for the analysis of temporal co-occurrences in a large, naturalistic movie dataset. This work was completed in Trinity College Dublin under the supervision of Prof. Rhodri Cusack.

## Motivation
Many large and curated datasets are unnaturalistic and contain specifically designed, esoteric subclasses. In order to better understand human cognition using models of vision and learning (e.g. deep neural networks) more naturalistic datasets are required.

In this work, we were interested in generating a visual dataset with preserved *temporal* structure. We chose 100 live action movies which contained typical scenarios and visual scenes that one may encounter throughout life. These movies were tagged using Amazon Web Services' Rekognition, and subsquent analyses examined the correlative structure of associated items across the entire dataset. We wanted to see whether semantically meaningful clusters would emerge from objects which occurred close together in time.

This analysis probed associations at a higher level than is typically examined in vision reasearch. Instead of testing the feature level clustering and association that occurs in object recognition, we aimed to investigate how objects relate to other objects in their surrounding context, and whether this signal can be learned to form more semantically meaningful category clusters.

## Data collection

[tag_movie.py](https://github.com/ClionaOD/associations/blob/master/data-collection/tag_movies.py) - this script sends movies which are located in an AWS S3 bucket to be automatically tagged using Amazon Rekognition. The program will return a pickle file to the specified s3 bucket with all labels and data for the requested video.

[create_itemsets.py](https://github.com/ClionaOD/associations/blob/master/data-collection/create_itemsets.py) - retrieve the raw output files from Rekognition (located in ./data) and process them into an array with each row containing item strings which occurred in the same 200 ms interval. 

**Summary of the dataset**

* 158.4 hr of live-action movies were tagged using Amazon Rekognition (mean movie length = 95.5 min).
* Labels were returned every 200 ms, resulting in 2,851,272 label arrays with 41,330,953 labels of which 2,466 were unique.
* This large array of categorical labels can be used to examine the time series of objects (whether they are present or absent) and to perform regression analyses on the various objects on view in the movies.

## Analysis

[temporal_regression.py](https://github.com/ClionaOD/associations/blob/master/temporal_regression.py) - perform Ridge Regression for the top most frequent items in the dataset of 40,000,000 labels. Retrieve the coefficient matrices for the diagonal and off diagonal values and plot their timecourses over a sweep from 200 ms to 2 hr. Plot the mean R2 values over an increasing number of lags from 0 ms to 2 hr.

[plotting.py](https://github.com/ClionaOD/associations/blob/master/plotting.py) - For chosen lags, plot the pairwise coefficient matrices for each of the 150 frequent items. 

## Results
Figures for various analyses are stored in [./results/figs](https://github.com/ClionaOD/associations/tree/master/results/figs)
