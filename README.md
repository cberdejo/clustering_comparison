# Clustering Multiple Techniques Comparisons
## Notebooks 

### [mall_customers](./notebooks/mall_customers.ipynb)

The dataset used in this study is the **Mall_Customers** from kaggle. Which is a simple dataset that contains demographic and spending information about a group of shopping mall clients. This dataset is widely used as a benchmark dataset in machine learning tutorials because of its small size, simplicity, and clear clustering structure. It is perfect for the introduction of the different clustering methods.

We performed different clustering analyses:

1. **K-Means Clustering**  
2. **Agglomerative (Hierarchical) Clustering**  
3. **DBSCAN (Density-Based Clustering)**  


### [forest_type.ipynb](./notebooks/forest_type.ipynb)

The dataset used in this study is the **Forest Type Mapping dataset** from the UCI Machine Learning Repository.  
It contains **523 samples**, each described by **27 numerical features**, and belongs to one of **4 classification classes** (forest types).  

The features include spectral band values and prediction errors from remote sensing data, which are highly correlated and reside in a relatively high-dimensional space.  
Because of this, directly visualizing or clustering the raw dataset is challenging.


To address the curse of dimensionality and improve interpretability, we applied **Principal Component Analysis (PCA)**:

- PCA transforms the 27 original variables into a smaller number of **principal components (PCs)** that capture the maximum variance in the data.  
- We reduced the dataset to **3 components**, which together retain most of the relevant variance.  
- This dimensionality reduction allows us to:
  - **Visualize the dataset in 3D** space while preserving key patterns.
  - **Facilitate clustering**, since noise and redundant variance are filtered out.
  - **Interpret variable importance** by analyzing the contribution of each feature to the PCs.

Once the data was projected into 3D PCA space, we performed different clustering analyses:

1. **K-Means Clustering**  
2. **Agglomerative (Hierarchical) Clustering**  
3. **DBSCAN (Density-Based Clustering)**  

### [human_activity_recognition.ipynb](./notebooks/human_activity_recognition.ipynb)

The dataset used in this study is the **Human Activity Recognition Using Smartphones** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).  

It contains accelerometer and gyroscope signals from smartphones worn by **30 subjects** performing **6 different activities** (walking, walking upstairs, walking downstairs, sitting, standing, laying).  
The dataset includes **561 features**, many of which are correlated and high-dimensional.  

To handle this complexity, we first applied **Principal Component Analysis (PCA)**, projecting the dataset into a lower-dimensional space that captures most of the variance.  
This makes the data easier to visualize (2D/3D) and more suitable for clustering.  

We then compared different clustering techniques:

1. **K-Means Clustering**  
   - Partitioned the PCA-transformed data into clusters by minimizing within-cluster variance.  
   - Evaluated using Silhouette Score, ARI, and NMI against the real activity labels.  

2. **Agglomerative (Hierarchical) Clustering**  
   - Tested multiple linkage criteria: Ward, Single, Complete, and Average.  
   - Compared how each linkage grouped the activities in PCA space.  
   - Ward linkage produced the most interpretable and compact clusters, closest to the real labels.  

3. **Self-Organizing Maps (SOM)**  
   - Applied SOM to project high-dimensional HAR data into a 2D grid, preserving the topological relationships between points.  
   - SOM provided an alternative visualization of the activity patterns and offered a competitive clustering approach compared to classical methods.  



##  Theoretical Framework
### Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a **dimensionality reduction technique** that transforms a dataset with possibly correlated features into a smaller set of uncorrelated variables, called *principal components*.  
These components are linear combinations of the original features, ordered by the amount of variance they capture:

- **First component**: captures the maximum variance in the data.
- **Second component**: orthogonal to the first, captures the next largest variance.
- And so on.

#### Why PCA is useful
- **Noise reduction**: by keeping only the most informative components, PCA filters out less relevant variance (often noise).
- **Visualization**: enables plotting high-dimensional datasets in 2D or 3D.
- **Preprocessing for clustering or prediction models**: reduces the curse of dimensionality, speeds up algorithms, and highlights hidden patterns.

PCA is not only a compression tool but also a **discovery method for latent structures** that may not be visible in the raw feature space.  
For clustering tasks, PCA often serves as a **preliminary step** to project the data into a space where clusters are more separable.

### Silhouette Score

The **Silhouette Score** is a general metric to evaluate clustering quality across all algorithms.  
It compares **cohesion** (how close points are to their own cluster) and **separation** (how far they are from other clusters):

- *a*: mean intra-cluster distance.  
- *b*: mean nearest-cluster distance.  
- Silhouette coefficient: `(b - a) / max(a, b)`  

Interpretation:
- **+1** → well-clustered point.  
- **0** → ambiguous, lies between clusters.  
- **<0** → possibly misclassified.  

By averaging the silhouette scores of all points, we obtain a global measure of clustering quality.  
This metric is often used to compare different clustering methods or to select the optimal number of clusters.




### Clustering Techniques

Clustering is an **unsupervised learning** approach aimed at grouping data points so that those in the same cluster are more similar to each other than to points in other clusters.  
It is often used as an exploratory step before predictive modeling, since it can reveal **hidden patterns and structure** in the data.

#### 1. K-Means Clustering
K-Means partitions data into **k clusters** by minimizing the variance within each cluster. It operates iteratively:
1. Randomly initialize k centroids.
2. Assign each point to the nearest centroid.
3. Update centroids as the mean of their assigned points.
4. Repeat until convergence.

- **Strengths**: simple, efficient, works well on spherical and equally sized clusters.
- **Limitations**: requires choosing *k* in advance; sensitive to outliers and cluster shapes.

##### Elbow Method
One common approach for estimating the optimal number of clusters is the **Elbow Method**.  
The procedure is:

1. Run K-Means for a range of *k* values (e.g., from 1 to 10).
2. Compute the **Within-Cluster Sum of Squares (WCSS)**, also called inertia, for each *k*.
3. Plot the number of clusters (*k*) against the WCSS.
4. Identify the point where the curve bends significantly, forming an **“elbow”** shape.

This “elbow” suggests a good trade-off: before the elbow, adding clusters drastically reduces WCSS; after the elbow, improvements are marginal. Thus, the elbow indicates the most suitable number of clusters.



#### 2. Agglomerative Clustering (Hierarchical)
Agglomerative clustering builds a **hierarchy of clusters** in a bottom-up fashion:
- Start with each data point as a single cluster.
- At each step, merge the two closest clusters.
- Continue until only one cluster remains (dendrogram representation).

The definition of "closest clusters" depends on the **linkage criterion**:

- **Ward linkage**: minimizes the variance within clusters (similar to K-Means philosophy). Tends to create compact, spherical clusters.
- **Single linkage**: distance between the nearest points of clusters. Prone to “chaining effect” (long, thin clusters).
- **Complete linkage**: distance between the farthest points. Produces compact clusters with similar diameters but may ignore natural structure.
- **Average linkage**: distance between clusters is defined as the average distance between all pairs of points across clusters. It provides a balance between single and complete linkage, reducing chaining while avoiding overly tight clusters, but may still produce less distinct boundaries than Ward.


- **Strengths**: no need to pre-specify the number of clusters; dendrogram gives flexibility.
- **Limitations**: computationally expensive for large datasets.



#### 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
DBSCAN groups points that are **closely packed** together, while labeling low-density points as **outliers**.  
It requires two parameters:
- **ε (epsilon)**: neighborhood radius.
- **minPts**: minimum number of points to form a dense region.

Process:
1. A point with at least *minPts* neighbors within ε is a **core point**.
2. Core points connect to form clusters.
3. Points within ε of a core point are **reachable**.
4. Points not reachable from any core point are labeled as **noise**.

- **Strengths**: discovers arbitrarily shaped clusters, robust to noise, does not require the number of clusters *k*.
- **Limitations**: sensitive to parameter choice; varying densities in the same dataset may cause issues.


#### 4. Self-Organizing Maps (SOM)
- A type of **artificial neural network** that projects high-dimensional data onto a lower-dimensional (usually 2D) grid.  
- Preserves **topological relationships**: points close in input space are mapped close in the SOM grid.  
- Useful for **visualization** and **clustering**, especially in high-dimensional datasets like HAR.  
- Produces a discrete mapping that highlights natural groupings in the data.  


