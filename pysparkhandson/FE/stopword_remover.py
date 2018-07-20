
from __future__ import print_function

# $example on$
from pyspark.ml.feature import StopWordsRemover
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("StopWordsRemoverExample")\
        .getOrCreate()

    # $example on$
    sentenceData = spark.createDataFrame([
        (0, ["I", "saw", "the", "red", "balloon"]),
        (1, ["Mary", "had", "a", "little", "lamb"])
    ], ["id", "raw"])

    remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
    remover.transform(sentenceData).show(truncate=False)
    # $example off$

    spark.stop()