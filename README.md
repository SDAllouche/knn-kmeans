# KNN and KMeans

## KNN

1. Inroduction

K Nearest Neighbours or KNN is the simplest of all machine learning algorithms, it falls under the Supervised Learning category and is used for classification (most commonly) and regression. In both classification and regression tasks, the input consists of the k closest training examples in the feature space. The output depends upon whether KNN is used for classification or regression purposes.

2. KNN intuition

The KNN algorithm intuition is very simple to understand. It simply calculates the distance between a sample data point and all the other training data points. The distance can be Euclidean distance or Manhattan distance. Then, it selects the k nearest data points where k can be any integer. Finally, it assigns the sample data point to the class to which the majority of the k data points belong.

3. KNN and Images

In this notebook we will see how to predict the class of a given image observation by identifying the observations that are nearest to it based on some features like color and texture.


![image](https://user-images.githubusercontent.com/102489525/231297565-c5d232e9-e8b0-4484-bd81-8c132d3fee88.jpg)


## KMeans

1. Inroduction

k-means is an unsupervised machine learning algorithm used to find groups of observations (Clustering) that share similar characteristics. What is the meaning of unsupervised learning? It means that the observations given in the data set are unlabeled, there is no outcome to be predicted.In other way k-means is used when we have unlabelled data which is data without defined categories or groups. The algorithm follows an easy or simple way to classify a given data set through a certain number of clusters, fixed apriori. K-Means algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity.

![kmeans](https://user-images.githubusercontent.com/102489525/231297713-b8b868c7-f567-4b21-a347-6bd5f1281900.png)

2. Application 

K-Means clustering is the most common unsupervised machine learning algorithm. It is widely used for many applications which include :
    - Image segmentation
    - Customer segmentation
    - Species clustering
    - Anomaly detection
    - Clustering languages

3. K-Means Clustering intuition

K-Means clustering is used to find intrinsic groups within the unlabelled dataset and draw inferences from them. It is based on centroid-based clustering.

Centroid - A centroid is a data point at the centre of a cluster. In centroid-based clustering, clusters are represented by a centroid. It is an iterative algorithm in which the notion of similarity is derived by how close a data point is to the centroid of the cluster. K-Means clustering works as follows.<br>The K-Means clustering algorithm uses an iterative procedure to deliver a final result. The algorithm requires number of clusters K and the data set as input. The data set is a collection of features for each data point. The algorithm starts with initial estimates for the K centroids. The algorithm then iterates between two steps:

   - Data assignment step : Each centroid defines one of the clusters. In this step, each data point is assigned to its nearest centroid, which is based on the squared Euclidean distance. So, if ci is the collection of centroids in set C, then each data point is assigned to a cluster based on minimum Euclidean distance.
    - Centroid update step : In this step, the centroids are recomputed and updated. This is done by taking the mean of all data points assigned to that centroid’s cluster.
    
The algorithm then iterates between step 1 and step 2 until a stopping criteria is met. Stopping criteria means no data points change the clusters, the sum of the distances is minimized or some maximum number of iterations is reached. This algorithm is guaranteed to converge to a result. The result may be a local optimum meaning that assessing more than one run of the algorithm with randomized starting centroids may give a better outcome.The K-Means intuition can be represented with the help of following diagram:

![kmeans2](https://user-images.githubusercontent.com/102489525/231297736-bbaec9fa-71ec-4a52-921e-ec446ed0e510.jpg)

4. Choosing the value of K

The K-Means algorithm depends upon finding the number of clusters and data labels for a pre-defined value of K. To find the number of clusters in the data, we need to run the K-Means clustering algorithm for different values of K and compare the results. So, the performance of K-Means algorithm depends upon the value of K. We should choose the optimal value of K that gives us best performance. There are different techniques available to find the optimal value of K. The most common technique is the elbow method (is used to determine the optimal number of clusters in K-means clustering. The elbow method plots the value of the cost function produced by different values of K. The below diagram shows how the elbow method works).

5. KNN and Images

In this notebook we will see how to find groups of a given images by identifying similar characteristics based on some features like color and texture.

## Image Descriptors

1. Color

The color is an important element in human perception because is helps us in the detection and differentiation of visual information and it's relatively simple to extract and compare.
Each pixel in an image contains information.To compare images, we need to extract their characteristics like :
- Medium Color 
- Color Histogram 
- Color Region

2. Texture is a similar structures repeated over and over again and distributed randomly. It provides information on the spatial distribution of colors by intensity in the image unlike Histograme.<br>
We going to use the statistical methods, that are based on the assumption that a texture is a realization of a two-dimensional stochastic process possessing the stationarity properties.

* GLCM( grey level co-occurrence matrics ) is a feature that allow to characterize the texture by several statistics. It measures the probability of occurrence of pixel value pairs located at a certain distance in the image.
It is based on the probability calculation P(i, j, δ, θ) which represents the number of times a color level pixel i appears at a relative distance δ from a color level pixel j and in a given orientation θ. The closest neighbours of pixel x in 4 directions are :

![Picture](https://user-images.githubusercontent.com/102489525/231298200-77e4f009-ac3c-4993-a004-e698f80a1e7a.jpg)

* There are 5 parameters :

Variance : It measures the heterogeneity of the texture.

Energy : This parameter measures the consistency of the texture.

Entropy : This parameter measures the disorder in the image.

Contrast : It measures the amount of variation of local intensities present in an image.

Homogeneity : It measures the homogeneity of the image.

## License

[MIT LICENSE](LICENSE)



