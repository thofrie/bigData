def preprocessing(df):
    df = df.drop(*['customernumber','date','domain','datecreated','title','points','advertisingdatacode','invoicepostcode','delivpostcode','deliverydatepromised','deliverydatereal'])
    features = df.schema.names
    for feature in features:
        df = df.withColumn(feature, df[feature].cast(IntegerType()))
    #df = df.drop('weight')
    df = df.withColumn("books",col("w0")+col("w1")+col("w2")+col("w3")+col("w4")+col("w5"))
    #df = df.drop(*["w0","w1","w2","w3","w4","w5","w6","w7","w8","w9","w10"])
    #df = df.withColumn("label",col("target90"))
    #df = df.drop("target90")
    return df

# Target in String?

def readData(data):
    data = data.map(lambda x: x.replace('"','')).persist()
    data = data.map(lambda x: x.replace(';;',';null;')).persist()
    data = data.map(lambda x: x.split(';')).persist()
    header = data.first()
    data = data.filter(lambda x: x != header).persist()
    df = sqlContext.createDataFrame(data, header)
    return df

def preprocessing(df):
    dfTrain = dfTrain.drop(*['date','domain','datecreated','title','points','advertisingdatacode','invoicepostcode','delivpostcode','deliverydatepromised','deliverydatereal'])
    features = dfTrain.schema.names
    for feature in features:
        dfTrain = dfTrain.withColumn(feature, dfTrain[feature].cast(IntegerType()))
    #dfTrain = dfTrain.drop('weight')
    dfTrain = dfTrain.withColumn("books",col("w0")+col("w1")+col("w2")+col("w3")+col("w4")+col("w5"))
    dfTrain = dfTrain.withColumn("nobooks",col("w6")+col("w7")+col("w8")+col("w9")+col("w10"))
    
    #dfTrain = dfTrain.drop(*["w0","w1","w2","w3","w4","w5","w6","w7","w8","w9","w10"])
    #dfTrain = dfTrain.withColumn("label",col("target90"))
    #dfTrain = dfTrain.drop("target90")
    return dfTrain
    
#the actual number of received items
(#ordered - (#canceled + #remitted))

dfTrain = dfTrain.drop(*['date','domain','datecreated','title','points','advertisingdatacode','invoicepostcode','delivpostcode','deliverydatepromised','deliverydatereal'])
features = dfTrain.schema.names
for feature in features:
    dfTrain = dfTrain.withColumn(feature, dfTrain[feature].cast(IntegerType()))
#dfTrain = dfTrain.drop('weight')
dfTrain = dfTrain.withColumn("books",col("w0")+col("w1")+col("w2")+col("w3")+col("w4")+col("w5"))
dfTrain = dfTrain.withColumn("nobooks",col("w6")+col("w7")+col("w8")+col("w9")+col("w10"))
dfTrain = dfTrain.withColumn("itemseff",col("numberitems")-(col("cancel") + col("remi")))

dfTest = dfTest.drop(*['date','domain','datecreated','title','points','advertisingdatacode','invoicepostcode','delivpostcode','deliverydatepromised','deliverydatereal'])
features = dfTest.schema.names
for feature in features:
    dfTest = dfTest.withColumn(feature, dfTest[feature].cast(IntegerType()))
#dfTest = dfTest.drop('weight')
dfTest = dfTest.withColumn("books",col("w0")+col("w1")+col("w2")+col("w3")+col("w4")+col("w5"))
dfTest = dfTest.withColumn("nobooks",col("w6")+col("w7")+col("w8")+col("w9")+col("w10"))
dfTest = dfTest.withColumn("itemseff",col("numberitems")-(col("cancel") + col("remi")))


def costMatrix(predicitionDF):
    tp = predictionDF[(predictionDF.label == 1) & (predictionDF.prediction == 1.0)].count()
    tn = predictionDF[(predictionDF.label == 0) & (predictionDF.prediction == 0.0)].count()
    fp = predictionDF[(predictionDF.label == 0) & (predictionDF.prediction == 1.0)].count()
    fn = predictionDF[(predictionDF.label == 1) & (predictionDF.prediction == 0.0)].count()
    return (tn * 1.5 - fn * 5) #revenue based on costMatrix

#tp = predictionDF[(predictionDF.label == 1) & (predictionDF.prediction == 1.0)].count()
#tn = predictionDF[(predictionDF.label == 0) & (predictionDF.prediction == 0.0)].count()
#fp = predictionDF[(predictionDF.label == 0) & (predictionDF.prediction == 1.0)].count()
#fn = predictionDF[(predictionDF.label == 1) & (predictionDF.prediction == 0.0)].count()
#revenue = tn * 1.5 - fn * 5


encoder0 = OneHotEncoder(inputCol='salutation', outputCol='salutation_Vec')
encoder1 = OneHotEncoder(inputCol='paymenttype', outputCol='paymenttype_Vec')
encoder2 = OneHotEncoder(inputCol='model', outputCol='model_Vec')
# Apply the encoder transformer
dfTrain = encoder0.transform(dfTrain)
dfTrain = encoder1.transform(dfTrain)
dfTrain = encoder2.transform(dfTrain)

encoder0 = OneHotEncoder(inputCol='salutation', outputCol='salutation_Vec')
encoder1 = OneHotEncoder(inputCol='paymenttype', outputCol='paymenttype_Vec')
encoder2 = OneHotEncoder(inputCol='model', outputCol='model_Vec')
# Apply the encoder transformer
dfTest = encoder0.transform(dfTest)
dfTest = encoder1.transform(dfTest)
dfTest = encoder2.transform(dfTest)

def encoding(df):
    encoder0 = OneHotEncoder(inputCol='salutation', outputCol='salutation_Vec')
    encoder1 = OneHotEncoder(inputCol='paymenttype', outputCol='paymenttype_Vec')
    encoder2 = OneHotEncoder(inputCol='model', outputCol='model_Vec')
    # Apply the encoder transformer
    df = encoder0.transform(df)
    df = encoder1.transform(df)
    df = encoder2.transform(df)
    return df

test_set = test_set.map(lambda x: x.replace('"','')).persist()
test_set = test_set.map(lambda x: x.replace(';;',';null;')).persist()
test_set = test_set.map(lambda x: x.split(';')).persist()
header = test_set.first()
test_set = test_set.filter(lambda x: x != header).persist()
dfTest = sqlContext.createDataFrame(test_set, header)

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

# RandomForest Feature Importance
print rf_model.featureImportances

# Convert feature importances to a pandas column
# & Convert list of feature names to pandas column
# sort DataFrame
fi_df = pd.DataFrame(rf_model.featureImportances.toArray(),
columns=['importance'])
fi_df['feature'] = pd.Series(allFeatures)
fi_df.sort_values(by=['importance'], ascending=False, inplace=True)

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
