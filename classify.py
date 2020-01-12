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
