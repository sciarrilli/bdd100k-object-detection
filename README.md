# BDD100k Object Detection

## Introduction

The BDD100k dataset is a large dataset of object detection annotations. This repo includes three different notebooks. A notebook for preprocessing the labels, a notebook for exploratory analysis and a notebook for modeling.

## Preprocessing

The BDD100k inclues a single large json file for train and a single json file for val labels. In order to not have to load the large json files into memory, we preprocess the labels into a multiple json files. 

## Exploratory Analysis

Exploring the dataset includes different histograms of the categories of objects, the number of objects per image, and the number of objects per category.

## Modeling

These notebooks include finetuning on a Faster R-CNN model. The most import part of the modeling is the customer dataset class which is used by pytorch dataloader. The target object is a dict with a few different keys: boxes, labels, and image_id are a few examples. The boxes key is a list of bounding boxes and their coordinates (x1, y1, x2, y2). The labels key is a list of objects detected per image. 

## Evaluation

Model evaluation is done by using the official pytorch evaluation script (engine.py). The best mAP@0.5:0.95 is 24.1%. 