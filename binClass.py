from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row

from pyspark.ml.param.shared import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler

confCluster = SparkConf().setAppName("BinaryClassification")
confCluster.set("spark.executor.memory", "8g")
confCluster.set("spark.executor.cores", "4")
repartition_count = 32
sc = SparkContext(conf=confCluster)
sqlContext = SQLContext(sc)


# Preprocess training data
# create RDD with training data 
train_set = sc.textFile("dmc2010_train.txt", minPartitions=repartition_count)
# each record is in " ", records have to be extracted with replace method
train_set = train_set.map(lambda x: x.replace('"','')).persist()
train_set = train_set.map(lambda x: x.replace(';;',';null;')).persist()
# extract features
train_set = train_set.map(lambda x: x.split(';')).persist()
# Count number of rows
# train_set.count()
# 32429 records with header
# create DataFrame of training data
# extract header of RDD
header = train_set.first()
train_set = train_set.filter(lambda x: x != header).persist()
# train_set.count() #32428
dfTrain = sqlContext.createDataFrame(train_set, header)
#len(dfTrain.columns) #38

# Count duplicates in training date 
#dfTrain.select(header).dropDuplicates().count()
# no duplicates found

# check DataTypes
dfTrain.printSchema()

# find Null values
dfTrain.select(['deliverydatepomised','deliverydatereal']).dropna().count()

### Feature Engineering drop useless columns
# List of columns to drop
cols_to_drop = ['customernumber','date','model','domain','datecreated','title','points','advertisingdatacode','invoicepostcode','delivpostcode','deliverydatepromised','deliverydatereal']
# Drop the columns
dfTrain = dfTrain.drop(*cols_to_drop)


# each variable is String, they have to be transformed
features = dfTrain.schema.names
for feature in features:
    dfTrain = dfTrain.withColumn(feature, dfTrain[feature].cast(IntegerType()))


# Summary statistics
dfTrain.describe().show()
summaryTrain = dfTrain.describe()
summaryTrain.toPandas().to_csv("summary.csv", encoding='utf-8')

### SQL
# Create a temporary table 
dfTrain.select('deliverydatepomised').createOrReplaceTempView("df")
# Construct a query to select the names of the people from the temporary table "people"
query = '''SELECT * FROM df Limit 10'''
# Assign the result of Spark's query to people_df_names
result_df = spark.sql(query)

rows = dfTrain.count()
summary = dfTrain.describe().filter(col("summary") == "count")
summary.select(*((lit(rows)-col(c)).alias(c) for c in dfTrain.columns)).show()

# Correlation
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
# Correlation Pearson 
# dfTrain.stat.corr('numberitems', 'weight') #ca. 0.77
dfTrain = dfTrain.drop('weight')

cardinalFeatures = ["numberitems","remi","cancel","used","w0","w1","w2","w3","w4","w5","w6","w7","w8","w9","w10"]

corrCoef = []
i = 0
#for feature in cardinalFeatures:
#    corrCoef.append(dfTrain.stat.corr(feature, 'target90'))

#correlation = sqlContext.createDataFrame(corrCoef, cardinalFeatures)

# Preprocess Test Data
# ...
dfTrain = dfTrain.withColumn("books",col("w0")+col("w1")+col("w2")+col("w3")+col("w4")+col("w5"))
dfTrain = dfTrain.withColumn("nobooks",col("w6")+col("w7")+col("w8")+col("w9")+col("w10"))