# Unsupervised vs Supervised — Mini-Project: PCA + K-Means on Iris Dataset

## Project Overview
This mini-project demonstrates the difference between unsupervised and supervised learning using the Iris dataset. The goal is to:

- Load a real-world dataset manually (Iris dataset from Kaggle).  
- Reduce the features from 4 dimensions to 2 using PCA (Principal Component Analysis).  
- Apply K-Means clustering on the PCA-reduced data.  
- If labels exist, map cluster IDs to actual species manually and compute the accuracy.  
- Visualize the clusters with centroids and save the final plot to a folder.  

This project helps understand how unsupervised methods (like K-Means) can try to discover patterns in data without labels, and how it compares to the actual labels (supervised information).

---

## Dataset
- **Name:** Iris Dataset  
- **Source:** Kaggle  

**Features (numeric):**  
- Sepal length  
- Sepal width  
- Petal length  
- Petal width  

**Target (labels):**  
- Setosa → 0  
- Versicolor → 1  
- Virginica → 2  

---

## Implementation Details

### 1. Loading the dataset
- I load the CSV manually using Python's `csv` module.  
- Labels are converted from strings (`setosa`, `versicolor`, `virginica`) to integers (0,1,2) for easier processing.  

### 2. PCA (Manual Implementation)
- Standardize the features manually using Z-score normalization.  
- Compute covariance matrix manually.  
- Use NumPy only to compute eigenvalues and eigenvectors.  
- Select the top 2 principal components to reduce features to 2D.  
- Project the standardized data onto these top 2 components manually.  

### 3. K-Means Clustering (Manual Implementation)
- Initialize centroids randomly.  
- Assign each data point to the closest centroid using Euclidean distance.  
- Update centroids by taking the mean of assigned points.  
- Repeat until centroids converge (no significant change).  

### 4. Mapping Clusters to Labels
- K-Means assigns arbitrary cluster IDs (0,1,2).  
- I manually map each cluster to the majority true label within that cluster.  
- Compute accuracy manually by comparing mapped cluster labels to true labels.  

### 5. Plotting Results
- Plot each cluster in 2D PCA space using different colors.  
- Centroids are plotted as black 'X'.  
- Save the plot to `plots/pca_kmeans_iris.png`.  

---

## How to Run
1. Place the `iris.csv` in the project folder.  
2. Run the script:

```bash
python run_pipeline.py
```
The pipeline will:

- Load the data

- Reduce dimensions using PCA

- Apply K-Means

- Compute accuracy

- Plot clusters and centroids

- Save the final plot in plots/ folder