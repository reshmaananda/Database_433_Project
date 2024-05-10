# Amazon Co-Purchase Recommendation Project
## Overview

This project aims to develop a copurchasing analysis using community detection and a recommendation system for Amazon co-purchase data using various machine learning algorithms and network analysis techniques. The system provides recommendations for items that are frequently purchased together based on historical co-purchase data from Amazon.

### Modules
- #### DataProcessing : 
      Python script for preprocessing raw Amazon co-purchase data. This includes cleaning, transforming, and preparing the data for analysis.
- #### Network Analysis:
      Performed network analysis on the preprocessed data. This includes creating a graph representation of the co-purchase network and calculating radius, diameter, density of the network and centrality metrics such as Betweeness centrality, closeness centrality, Degree Centrality, Eigenvector centrality, PageRank

       We detected communities using Girvan-newman and Louvain community detection algorithim 
- #### Link Prediction & Machine Learning
      For predicting future co-purchases between items in the network. This includes implementing algorithms such as GNN (Graph Neural Networks) and traditional machine learning algorithms like Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine.
- #### Book Recommendation System
       For generating book recommendations based on the co-purchase network analysis and link prediction results.

## Features



## Dependencies

- Python 3.x
- NetworkX
- Pandas
- Numpy
- torch_geometric
- Scikit-learn
- Matplotlib (for visualization)

## Contributors
- [Reshma Ananda Prabhakar](https://github.com/reshmaananda/)
- [	Hafeeza Begum Dudekula](https://github.com/HafeezaBegum)
- [Lokesh Poluru Velayudham](https://github.com/lokeshvelayudham/)

## Results
Co purchasing analysis using Louvain and Girvan newman algorithim detected all possible communities and the recommendation system provides accurate book recommendations based on the co-purchase behavior of Amazon customers. Evaluation metrics such as accuracy, MSE, ROC AUC Score.


