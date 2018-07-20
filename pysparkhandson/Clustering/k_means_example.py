

from __future__ import print_function

# $example on$
from numpy import array
from math import sqrt
# $example off$

from pyspark import SparkContext
# $example on$
from pyspark.mllib.clustering import KMeans, KMeansModel
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="KMeansExample")  # SparkContext

    # $example on$
    # Load and parse the data
    data = sc.textFile("file:///opt/hadoop/spark-ml/Clustering/kmeans_data.txt")
    parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

    # Build the model (cluster the data)
    clusters = KMeans.train(parsedData, 2, maxIterations=100, initializationMode="random")


    print(clusters.predict([0.0 , 1.0 , 3.0]))
    print(clusters.predict([0.0 , 0.0 , 0.0]))    
    print (clusters.clusterCenters)



    sc.stop()
