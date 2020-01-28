# einlesen in eine Funktion
# function one hot encoding

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

# Cluster Configuration
confCluster = SparkConf().setAppName("BinaryClassification")
confCluster.set("spark.executor.memory", "8g")
confCluster.set("spark.executor.cores", "4")
repartition_count = 32
sc = SparkContext(conf=confCluster)
sqlContext = SQLContext(sc)

#### Preprocessing
start_time = time()

#############################################
# Preprocess test data has same structure as train data, but without target variable
train_set = sc.textFile("dmc2010_train.txt", minPartitions=repartition_count)
test_set = sc.textFile("dmc2010_class.txt", minPartitions=repartition_count)

def transformData(dataset):
    dataset = dataset.map(lambda x: x.replace('"','')).persist()
    dataset = dataset.map(lambda x: x.replace(';;',';null;')).persist()
    dataset = dataset.map(lambda x: x.split(';')).persist()
    header = dataset.first()
    dataset = dataset.filter(lambda x: x != header).persist()
    df = sqlContext.createDataFrame(dataset, header)
    return df

dfTrain = transformData(train_set)
dfTest = transformData(test_set)

#############################################
def preprocessing(df):
    df = df.drop(*['domain','title','points','advertisingdatacode','invoicepostcode','delivpostcode','deliverydatepromised','deliverydatereal'])
    
    df = df.withColumn("books",col("w0")+col("w1")+col("w2")+col("w3")+col("w4")+col("w5"))
    df = df.withColumn("nobooks",col("w6")+col("w7")+col("w8")+col("w9")+col("w10"))
    df = df.withColumn("itemseff",col("numberitems")-(col("cancel") + col("remi")))
    df = df.withColumn("date", df.date.cast(DateType()))
    df = df.withColumn("datecreated", df.datecreated.cast(DateType()))
    df = df.withColumn("accountdur", datediff(col('date'),col('datecreated')))
    df = df.drop(*['date','datecreated'])
    features = df.schema.names
    for feature in features:
        df = df.withColumn(feature, df[feature].cast(IntegerType()))

    return df
#############################################

dfTrain = preprocessing(dfTrain)
dfTest = preprocessing(dfTest)
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
dfTrain = dfTrain.unionAll(dfTrain_under_sample) # combine dataframes to oversampled dataframe
# dfTrain.filter(dfTrain.label == 0).count() #26377
# dfTrain.filter(dfTrain.label == 1).count() #26671
# now we have equally distributed class labels

# Feature Engineering (Selection)
# OneHot Encoding of nominal scaled features
def encoding(df):
    encoder0 = OneHotEncoder(inputCol='salutation', outputCol='salutation_Vec')
    encoder1 = OneHotEncoder(inputCol='paymenttype', outputCol='paymenttype_Vec')
    encoder2 = OneHotEncoder(inputCol='model', outputCol='model_Vec')
    # Apply the encoder transformer
    df = encoder0.transform(df)
    df = encoder1.transform(df)
    df = encoder2.transform(df)
    return df

# Apply one hot encoding 
dfTrain = encoding(dfTrain) 
dfTest = encoding(dfTest)

end_time = time()
elapsed_time = end_time - start_time
print("Time of preprocessing: %.3f seconds" % elapsed_time)
#### End of preprocessing

#### Modeling & Evaluation
# Evaluation criterion is generated revenue based on a given cost matrix 
def costMatrix(predicitionDF):
    #tp = predictionDF[(predictionDF.label == 1) & (predictionDF.prediction == 1.0)].count()
    tn = predictionDF[(predictionDF.label == 0) & (predictionDF.prediction == 0.0)].count()
    #fp = predictionDF[(predictionDF.label == 0) & (predictionDF.prediction == 1.0)].count()
    fn = predictionDF[(predictionDF.label == 1) & (predictionDF.prediction == 0.0)].count()
    return (tn * 1.5 - fn * 5) #revenue based on costMatrix

# Training of a Decision Tree model with all Features
# dfTrain = dfTrain.drop("features") 
# dfTest = dfTest.drop("features")

# selected features of preprocessed datasets for modeling
allFeatures = ['salutation_Vec','newsletter','model_Vec','paymenttype_Vec','voucher','case','numberitems','gift','entry','shippingcosts','weight','remi','cancel','used','w0','w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','books','nobooks','itemseff']
assembler = VectorAssembler(inputCols=allFeatures, outputCol='features')
dfTrain = assembler.transform(dfTrain)
dfTest = assembler.transform(dfTest)

# Training + Prediction
###################################

revenues = [] # List of generated results

#### Decision Tree Classifier
start_time = time()
tree = DecisionTreeClassifier()
tree_model = tree.fit(dfTrain)
predictionTree = tree_model.transform(dfTest)
end_time = time()
elapsed_time = end_time - start_time
print("Time to train Tree on dfTrain and make predictions on dfTest: %.3f seconds" % elapsed_time)

# Evaluation
predictionTree.groupBy("label", "prediction").count().show()
start_time = time()
#revenue = costMatrix(predictionTree)
tn = predictionTree[(predictionTree.label == 0) & (predictionTree.prediction == 0.0)].count()
fn = predictionTree[(predictionTree.label == 1) & (predictionTree.prediction == 0.0)].count()
revenue = (tn * 1.5 - fn * 5) #revenue based on costMatrix
revenues.append(revenue)
end_time = time()
elapsed_time = end_time - start_time

print("Time to evaluate model: %.3f seconds" % elapsed_time)

# further metrics for evaluation of the classifier
""" evaluatorM = MulticlassClassificationEvaluator()
evaluatorM.evaluate(predictionTree, {evaluatorM.metricName: 'accuracy'}) 
evaluator = BinaryClassificationEvaluator()
print("Tree: Test_SET (Area Under ROC): " + str(evaluator.evaluate(predictionTree, {evaluator.metricName: "areaUnderROC"})))
print("Tree: Test_SET (Area Under PR): " + str(evaluator.evaluate(predictionTree, {evaluator.metricName: "areaUnderPR"})))
evaluator = BinaryClassificationEvaluator() """

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
tn = predictionRF[(predictionRF.label == 0) & (predictionRF.prediction == 0.0)].count()
fn = predictionRF[(predictionRF.label == 1) & (predictionRF.prediction == 0.0)].count()
revenue = (tn * 1.5 - fn * 5) #revenue based on costMatrix
revenues.append(revenue)
end_time = time()
elapsed_time = end_time - start_time
print("Time to evaluate model: %.3f seconds" % elapsed_time)

# further metrics for evaluation of the classifier
""" evaluatorM = MulticlassClassificationEvaluator()
evaluatorM.evaluate(predictionTree, {evaluatorM.metricName: 'accuracy'}) 
evaluator = BinaryClassificationEvaluator()
print("Tree: Test_SET (Area Under ROC): " + str(evaluator.evaluate(predictionRF, {evaluator.metricName: "areaUnderROC"})))
print("Tree: Test_SET (Area Under PR): " + str(evaluator.evaluate(predictionRF, {evaluator.metricName: "areaUnderPR"})))
 """

#####################################################################################
#### Backward Feature Elimination

feature_list_global = ['salutation_Vec','newsletter','model_Vec','paymenttype_Vec','voucher','case','numberitems','gift','entry','shippingcosts','weight','remi','cancel','used','w0','w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','books','nobooks','itemseff']
dropList = []
revenues=[]
maxRevenues=[]
meanTimes = []
times = []
features_remain = feature_list_global[:]

#tree = DecisionTreeClassifier()
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label',maxDepth = 5)

""" for j in range(len(feature_list_global)):
    for i in range(len(features_remain)):
        
        features_iter = features_remain[:]

        dfTrain = dfTrain.drop("features")
        dfTest = dfTest.drop("features")


        if(len(features_iter)>1):
            del features_iter[i]

        assembler = VectorAssembler(inputCols=features_iter, outputCol='features')
        dfTrain = assembler.transform(dfTrain)
        dfTest = assembler.transform(dfTest)
        
        start_time = time()
        #tree_model = tree.fit(dfTrain)
        #predictionTree = tree_model.transform(dfTest)
        rf_model = rf.fit(dfTrain)
        predictionRF = rf_model.transform(dfTest)
        end_time = time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        #tn = predictionTree[(predictionTree.label == 0) & (predictionTree.prediction == 0.0)].count()
        #fn = predictionTree[(predictionTree.label == 1) & (predictionTree.prediction == 0.0)].count()
        tn = predictionRF[(predictionRF.label == 0) & (predictionRF.prediction == 0.0)].count()
        fn = predictionRF[(predictionRF.label == 1) & (predictionRF.prediction == 0.0)].count()
        revenue = (tn * 1.5 - fn * 5) #revenue based on costMatrix
        revenues.append(revenue)
        print(features_iter)
        print("____________________________")
    
    meanTimes.append(sum(times)/len(times))
    times = [] # clear times
    maxRevenues.append(max(revenues))
    index_dropped = revenues.index(max(revenues)) # max Revenue without Feature at this index
    dropList.append(features_remain[index_dropped])
    del features_remain[index_dropped]# update of features_remain
    revenues=[]

revenuePDF = pd.DataFrame(maxRevenues,columns=["maxRevenues"])
revenuePDF["timeRF"] = meanTimes
revenuePDF["dropped"] = dropList
revenuePDF.to_csv("binClassRF.csv", encoding='utf-8') 
 """
#####################################################################################

#summaryTrain.toPandas().to_csv("summary.csv", encoding='utf-8')

##### Data Understanding
# Correlation matrix between cardinal features
""" cardinalFeatures = ['numberitems','weight','remi','cancel','used','w0','w1','w2','w3','w4','w5','w6','w7','w8','w9','w10','books','nobooks','itemseff']
assembler = VectorAssembler(inputCols=cardinalFeatures, outputCol="vector_cardinal")
df_cardinal = assembler.transform(dfTrain).select("vector_cardinal")
matrix = Correlation.corr(df_cardinal, "vector_cardinal")
matrix.collect()[0]["pearson({})".format("vector_cardinal")].values
 """
dfTrain.unpersist()
dfTest.unpersist()
dfTrain_under_sample.unpersist()
dfTrain_under.unpersist()

