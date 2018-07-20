from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import NaiveBayes   

from pyspark.mllib.evaluation import BinaryClassificationMetrics
if __name__ == "__main__":
	sc = SparkContext(appName="LogisticReg")
	spam = sc.textFile("file:///opt/hadoop/spark-ml/Classification/spam.txt")
	normal = sc.textFile("file:///opt/hadoop/spark-ml/Classification/nospam.txt")
	# Create a HashingTF instance to map email text to vectors of 10,000 features.
	tf = HashingTF(numFeatures = 50000)
	# Each email is split into words, and each word is mapped to one feature.
	spamFeatures = spam.map(lambda email: tf.transform(email.split(" ")))
	normalFeatures = normal.map(lambda email: tf.transform(email.split(" ")))
	# Create LabeledPoint datasets for positive (spam) and negative (normal) examples.
	positiveExamples = spamFeatures.map(lambda features: LabeledPoint(1, features))
	negativeExamples = normalFeatures.map(lambda features: LabeledPoint(0, features))
	trainingData = positiveExamples.union(negativeExamples)
	#trainingData.cache() # Cache since Logistic Regression is an iterative algorithm.
	
	training, test = trainingData.randomSplit([0.8, 0.2])


	# Run Logistic Regression using the SGD algorithm.
	model = LogisticRegressionWithSGD.train(trainingData)
	#	model = NaiveBayes.train(trainingData)
	#model = NaiveBayes.train(training)
	
	predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))

	metrics = BinaryClassificationMetrics(predictionAndLabels)

	# Overall statistics
	#print("Recall = %s" % metrics.recall())
#	#print("Precision = %s" % metrics.precision())
	print("Area Under PR = %s" % metrics.areaUnderPR)
	print("AUC = %s" % metrics.areaUnderROC)
	#print (metrics.confusionMatrix().toArray())
	# Test on a positive example (spam) and a negative one (normal). We first apply
	# the same HashingTF feature transformation to get vectors, then apply the model.
	posTest = tf.transform("To find out who it is call from a landline 09111032124".split(" "))
	negTest = tf.transform("Hi Dad, I started studying Spark the other ...".split(" "))
	print ("Prediction for positive test example: %g" % model.predict(posTest))
	print ("Prediction for negative test example: %g" % model.predict(negTest))
