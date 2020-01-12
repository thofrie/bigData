from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.types import DateType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.functions import datediff
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder, StringIndexer

from pyspark.ml.param.shared import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator


from time import *
import pandas as pd

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
# extract header of RDD
header = train_set.first()
train_set = train_set.filter(lambda x: x != header).persist()
# train_set.count() #32428
dfTrain = sqlContext.createDataFrame(train_set, header)

#############################################
# Preprocess test data: same structure, but without target variable
test_set = sc.textFile("dmc2010_class.txt", minPartitions=repartition_count)
test_set = test_set.map(lambda x: x.replace('"','')).persist()
test_set = test_set.map(lambda x: x.replace(';;',';null;')).persist()
test_set = test_set.map(lambda x: x.split(';')).persist()
header = test_set.first()
test_set = test_set.filter(lambda x: x != header).persist()
dfTest = sqlContext.createDataFrame(test_set, header)
# cast Label to integer?
#############################################
def preprocessing(df):
    df = df.drop(*['date','domain','datecreated','title','points','advertisingdatacode','invoicepostcode','delivpostcode','deliverydatepromised','deliverydatereal'])
    features = df.schema.names
    for feature in features:
        df = df.withColumn(feature, df[feature].cast(IntegerType()))
    #df = df.drop('weight')
    df = df.withColumn("books",col("w0")+col("w1")+col("w2")+col("w3")+col("w4")+col("w5"))
    df = df.withColumn("nobooks",col("w6")+col("w7")+col("w8")+col("w9")+col("w10"))
    df = df.withColumn("itemseff",col("numberitems")-(col("cancel") + col("remi")))
    #df = df.drop(*["w0","w1","w2","w3","w4","w5","w6","w7","w8","w9","w10"])
    #df = df.withColumn("label",col("target90"))
    #df = df.drop("target90")
    #df = df.withColumn("deliverydatepromised", df.deliverydatepromised.cast(DateType()))
    #df = df.withColumn("deliverydatereal", df.deliverydatereal.cast(DateType()))
    #df = df.withColumn("deliverydiff", datediff(col('deliverydatepromised'),col('deliverydatereal')))
    #date überzogen
    return df
#############################################

#dfTrain = preprocessing(dfTrain)
#dfTest = preprocessing(dfTest)
dfTrain = dfTrain.withColumn("label",col("target90")) #rename target variable for classifier, get right index
dfTrain = dfTrain.drop("target90") #drop old column


# get labels of test data
# labels are in external file, they have to be joined to the dfTest DataFrame
classLabels = sc.textFile("dmc2010_real.txt", minPartitions=repartition_count)
classLabels = classLabels.map(lambda x: x.replace('"','')).persist()
classLabels = classLabels.map(lambda x: x.replace(';;',';null;')).persist()
classLabels = classLabels.map(lambda x: x.split(';')).persist()
dfClassLabel = sqlContext.createDataFrame(classLabels, ["customernumber","label"])
dfTest = dfTest.join(dfClassLabel, dfTest.customernumber == dfClassLabel.customernumber,how='inner')
dfTest.drop("customernumber")
dfTest = dfTest.withColumn("label", dfTest["label"].cast(IntegerType()))

# imbalanced training data, assign weights to classes
# without Rebalancing: Metric Trap: Tree has accuracy of ca. 81 %
# --> 81 % has Label 0, Tree predicts always 0
# penalize majority class by assigning less weight, boost minority class with bigger weight
# get distribution ration in traing data
# ratio = float(dfTrain.filter(dfTrain.label == 0).count()) / float(dfTrain.count()) # 0.813
# ratio = 0.813
# dfTrain = dfTrain.withColumn('weights', F.when(col("label") == 1, ratio).otherwise(1-ratio)).persist() # apply weight balance
# dfTest = dfTest.withColumn('weights', F.when(col("label") == 1, ratio).otherwise(1-ratio)).persist()
# !!!Problem: Algorithms in pyspark don't have hyperparameter for classweights

# Solution: Oversampling of minority class
fraction = float(dfTrain.filter(dfTrain.label == 0).count()) / float(dfTrain.filter(dfTrain.label == 1).count()) # ratio of zeros and ones
fraction = fraction - 2 # because dataframe has already 6051 values of 1

dfTrain_under = dfTrain.filter(dfTrain.label == 1) # get dataframe of underrepresentated class
dfTrain_under_sample = dfTrain_under.sample(withReplacement=True, fraction = fraction, seed = 0)
dfTrain_over = dfTrain.unionAll(dfTrain_under_sample) # combine dataframes to oversampled dataframe
# dfTrain.filter(dfTrain.label == 0).count() #26377
# dfTrain.filter(dfTrain.label == 1).count() #26671
# now we have equally distributed class labels
dfTrain = dfTrain.unionAll(dfTrain_under_sample)
# Feature Engineering (Selection)

# One Hot Encoding
# Create encoder transformer
encoder0 = OneHotEncoder(inputCol='salutation', outputCol='salutation_Vec')
encoder1 = OneHotEncoder(inputCol='paymenttype', outputCol='paymenttype_Vec')
encoder2 = OneHotEncoder(inputCol='model', outputCol='model_Vec')
# Apply the encoder transformer
dfTrain = encoder0.transform(dfTrain)
dfTrain = encoder1.transform(dfTrain)
dfTrain = encoder2.transform(dfTrain)


#### Modeling & Evaluation
#allFeatures = ['salutation_Vec','newsletter','model_Vec','paymenttype_Vec','voucher','case','numberitems','gift','entry','shippingcosts','weight','remi','cancel','used','w0','w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','books','nobooks','itemseff']
#assembler = VectorAssembler(inputCols=['case','model_Vec','paymenttype_Vec','salutation_Vec','newsletter', 'voucher','books','nobooks','numberitems','itemseff'], outputCol='features')
def costMatrix(predicitionDF):
    tp = predictionDF[(predictionDF.label == 1) & (predictionDF.prediction == 1.0)].count()
    tn = predictionDF[(predictionDF.label == 0) & (predictionDF.prediction == 0.0)].count()
    fp = predictionDF[(predictionDF.label == 0) & (predictionDF.prediction == 1.0)].count()
    fn = predictionDF[(predictionDF.label == 1) & (predictionDF.prediction == 0.0)].count()
    return (tn * 1.5 - fn * 5) #revenue based on costMatrix

revenuesTree = []

# Training of a Decision Tree model with all Features
allFeatures = ['salutation_Vec','newsletter','model_Vec','paymenttype_Vec','voucher','case','numberitems','gift','entry','shippingcosts','weight','remi','cancel','used','w0','w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','books','nobooks','itemseff']
assembler = VectorAssembler(inputCols=allFeatures, outputCol='features')
dfTrain = dfTrain.drop("features") 
dfTest = dfTest.drop("features")
dfTrain = assembler.transform(dfTrain)
dfTest = assembler.transform(dfTest)

# Training + Prediction
start_time = time()
tree = DecisionTreeClassifier(maxDepth=5)
tree_model = tree.fit(dfTrain)
predictionTree = tree_model.transform(dfTest)
end_time = time()
elapsed_time = end_time - start_time
print("Time to train Tree on dfTrain and make predictions on dfTest: %.3f seconds" % elapsed_time)

# Evaluation
predictionTree.groupBy("label", "prediction").count().show()
start_time = time()
revenue = costMatrix(predictionTree)
end_time = time()
elapsed_time = end_time - start_time
print(revenue)
print("Time to evaluate model: %.3f seconds" % elapsed_time)

###################################
# Random Forest with all Features
# Training + Prediction
start_time = time()
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rf_model = rf.fit(dfTrain)
predictionRF = rf_model.transform(dfTest)
end_time = time()
elapsed_time = end_time - start_time
print("Time to train Random Forest on dfTrain and make predictions on dfTest: %.3f seconds" % elapsed_time)

# Evaluation
predictionRF.groupBy("label", "prediction").count().show()
start_time = time()
revenue = costMatrix(predictionRF)
end_time = time()
elapsed_time = end_time - start_time
print(revenue)
print("Time to evaluate model: %.3f seconds" % elapsed_time)

# RandomForest Feature Importance
print rf_model.featureImportances

# Convert feature importances to a pandas column
# & Convert list of feature names to pandas column
# sort DataFrame
fi_df = pd.DataFrame(rf_model.featureImportances.toArray(),
columns=['importance'])
fi_df['feature'] = pd.Series(allFeatures)
fi_df.sort_values(by=['importance'], ascending=False, inplace=True)


#revenuesTree.append(revenue)

# Metrics for evaluation of the classifier
evaluatorM = MulticlassClassificationEvaluator()
evaluatorM.evaluate(predictionTree, {evaluatorM.metricName: 'accuracy'}) 
evaluator = BinaryClassificationEvaluator()
print("Tree: Test_SET (Area Under ROC): " + str(evaluator.evaluate(predictionTree, {evaluator.metricName: "areaUnderROC"})))
print("Tree: Test_SET (Area Under PR): " + str(evaluator.evaluate(predictionTree, {evaluator.metricName: "areaUnderPR"})))
evaluator = BinaryClassificationEvaluator()


#summaryTrain.toPandas().to_csv("summary.csv", encoding='utf-8')


# Count number of rows
# train_set.count()
# 32429 records with header
# create DataFrame of training data
# extract header of RDD
# train_set.count() #32428
dfTrain.filter(dfTrain.label == 1).count() #6051
dfTrain.filter(dfTrain.label == 0).count() #26377
dfTest.filter(dfTest.label == 1).count() #6168
dfTest.filter(dfTest.label == 0).count() #26259


dfTrain = sqlContext.createDataFrame(train_set, header)
#len(dfTrain.columns) #38

# Count duplicates in training date 
#dfTrain.select(header).dropDuplicates().count()
# no duplicates found

# check DataTypes
dfTrain.printSchema()

# find Null values
dfTrain.select(['deliverydatepomised','deliverydatereal']).dropna().count()



# Summary statistics
dfTrain.describe().show()
summaryTrain = dfTrain.describe()
summaryTrain.toPandas().to_csv("summary.csv", encoding='utf-8')



# Correlation
# Correlation Pearson 
# dfTrain.stat.corr('numberitems', 'weight') #ca. 0.77

cardinalFeatures = ["numberitems","remi","cancel","used","w0","w1","w2","w3","w4","w5","w6","w7","w8","w9","w10"]
corrCoef = []
i = 0
#for feature in cardinalFeatures:
#    corrCoef.append(dfTrain.stat.corr(feature, 'target90'))

#correlation = sqlContext.createDataFrame(corrCoef, cardinalFeatures)
