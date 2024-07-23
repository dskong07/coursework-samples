import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR

# Import your own dependencies

from pyspark.sql import functions as F


from pyspark.sql.functions import variance

from pyspark.sql.functions import isnan




from pyspark.sql.functions import explode

from pyspark.sql.functions import col

from pyspark.sql.functions import size

from pyspark.sql.functions import explode_outer

#from pyspark.sql.DataFrame import approxQuantile

from pyspark.sql import DataFrameStatFunctions as stat

from pyspark.ml.feature import StringIndexer, OneHotEncoder

from pyspark.ml.evaluation import RegressionEvaluator

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics


# ---------- Begin definition of helper functions, if you need any ------------

# def task_1_helper():
#   pass

# -----------------------------------------------------------------------------


def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    #     joined_prod_rev = product_data.join(review_data, on="asin",how="left")
    
#     mean_rating = joined_prod_rev.groupBy('asin').mean()  #.avg()
#     count_ratings = review_data.groupBy('asin').count()
    temp = review_data.groupby("asin").agg(F.avg("overall").alias("meanRating"),
                                          F.count("overall").alias("countRating"))
    
    output = product_data.join(temp, on="asin",how="left")
    
    
 
    
    #mean_rating.show(20)
    #count_ratings.show(20)
    
#     mean_rating = joined_prod_rev.groupby("asin").agg(F.avg("overall").alias("meanRating"))
    
#     count_ratings = joined_prod_rev.groupby("asin").agg(F.count("overall").alias("countRating"))

#   mean_rating.show(20)
#   count_ratings.show(20)
    
    #print(mean_rating.count())
    #print(count_ratings.count())





    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:
    
    res['count_total'] = output.count()
    
    res['mean_meanRating'] = output.agg(F.avg('meanRating')).collect()[0][0]
    
    res['variance_meanRating'] = output.agg({'meanRating': 'variance'}).collect()[0][0]
    
    res['numNulls_meanRating'] = output.select(F.count(F.when(F.isnull(F.col("meanRating")), 1))).collect()[0][0]
    
    
    res['mean_countRating'] = output.select(F.avg(F.col('countRating')).alias('out1')).collect()[0]['out1']
    
    res['variance_countRating'] = output.agg({'countRating': 'variance'}).collect()[0][0]
    
    res['numNulls_countRating'] = output.select(F.count(F.when(F.isnull(F.col("countRating")), 1))).collect()[0][0]
    




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    
    prod_data = product_data.withColumn('category', F.col('categories')[0][0])
    
    prod_data = prod_data.withColumn('category', F.when(col('category') == '', None).otherwise(col('category')))
    
    
    
    
    intermediate_df = prod_data.select(F.col(category_column), 
                                       F.explode_outer(col(salesRank_column))
                                           .alias(bestSalesCategory_column, bestSalesRank_column))
    
    ct = intermediate_df.count()
    
    mean_bestSalesRank = intermediate_df.select(F.avg(bestSalesRank_column)).collect()[0][0]
    
    variance_bestSalesRank = intermediate_df.select(F.variance(bestSalesRank_column)).collect()[0][0]
    
    numNulls_category = intermediate_df.filter("category IS NULL").count()
    
    
    
    countDistinct_category = intermediate_df.select(F.countDistinct(category_column)).collect()[0][0]
    
    numNulls_bestSalesCategory = intermediate_df.filter("bestSalesCategory IS NULL").count()
    
    countDistinct_bestSalesCategory = intermediate_df.select(F.countDistinct(bestSalesCategory_column)).collect()[0][0]





    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': ct,
        'mean_bestSalesRank': mean_bestSalesRank,
        'variance_bestSalesRank': variance_bestSalesRank,
        'numNulls_category': numNulls_category,
        'countDistinct_category': countDistinct_category,
        'numNulls_bestSalesCategory': numNulls_bestSalesCategory,
        'countDistinct_bestSalesCategory':countDistinct_bestSalesCategory
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    df = product_data.select(asin_column, F.explode_outer(col(related_column)).alias("related_keys", "id_list"))
    
    
    df2 = df.select(asin_column, 'related_keys', F.explode_outer(col('id_list')))
    
    
    prices = product_data.select(asin_column, price_column).withColumnRenamed('asin', 'asin2')
    merged = df2.join(prices, prices.asin2 == df2.col, "leftouter") 
    
   
    merged = merged.withColumn('price', F.when(col('related_keys') != 'also_viewed', None).otherwise(col('price')))
    
    
    grouped = merged.groupby(asin_column).agg(F.mean(price_column).alias('meanPriceAlsoViewed'))
    grouped_ct = grouped.count()
    
    
    mean_meanPriceAlsoViewed = grouped.select(F.avg('meanPriceAlsoViewed')).collect()[0][0]
    variance_meanPriceAlsoViewed = grouped.select(F.variance('meanPriceAlsoViewed')).collect()[0][0]
    numNulls_meanPriceAlsoViewed = grouped.filter(F.col("meanPriceAlsoViewed").isNull()).count()
    
    
    
    
    df3 = df = df.withColumn('related_keys', F.when(col('related_keys') !='also_viewed', None).otherwise(col('related_keys')))
    df3 = df3.withColumn('id_list', F.when(col('related_keys').isNull(), None).otherwise(col('id_list')))
    df3 = df3.select('*', size('id_list').alias('product_cnt'))
    df3 = df3.withColumn('product_cnt', F.when(col('id_list').isNull(), None).otherwise(col('product_cnt')))
    
    
    grouped2 = df3.groupby(asin_column).agg(F.mean('product_cnt').alias('countAlsoViewed'))

    
    mean_countAlsoViewed = grouped2.select(F.avg('countAlsoViewed')).collect()[0][0]
    
    variance_countAlsoViewed = grouped2.select(F.variance('countAlsoViewed')).collect()[0][0]
    
    numNulls_countAlsoViewed = grouped2.filter(F.col("countAlsoViewed").isNull()).count()




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': grouped_ct,
        'mean_meanPriceAlsoViewed': mean_meanPriceAlsoViewed,
        'variance_meanPriceAlsoViewed': variance_meanPriceAlsoViewed,
        'numNulls_meanPriceAlsoViewed': numNulls_meanPriceAlsoViewed,
        'mean_countAlsoViewed': mean_countAlsoViewed,
        'variance_countAlsoViewed': variance_countAlsoViewed,
        'numNulls_countAlsoViewed': numNulls_countAlsoViewed
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    product_data = product_data.withColumn('price', product_data.price.cast('float'))
    
    total_count = product_data.count()
    
    df = product_data.filter("price IS NOT NULL")
    
    imputed_mean = df.select(F.avg('price')).collect()[0][0]
    
    imputed_median = df.stat.approxQuantile(price_column, [0.5], 0.01)[0]
    
    df1 = product_data.na.fill(value=imputed_mean, subset=['price'])
    
    meanOf_meanImputedPrice = df1.select(F.avg(price_column)).collect()[0][0]
    
    varOf_meanImputedPrice = df1.select(F.variance(price_column)).collect()[0][0]
    
    numNullsOf_meanImputedPrice = df1.select([F.count(F.when(isnan(c) | col(c).isNull(), c)).alias(c) 
                                              for c in ['price']]).collect()[0][0]
    
    
    
    
    df2 = product_data.na.fill(value=imputed_median, subset=['price'])
    
    meanOf_medianImputedPrice = df2.select(F.avg(price_column)).collect()[0][0]
    
    varOf_medianImputedPrice = df2.select(F.variance(price_column)).collect()[0][0]
    
    numNullsOf_medianImputedPrice = df2.select([F.count(F.when(isnan(c) | col(c).isNull(), c)).alias(c)
                                               for c in ['price']]).collect()[0][0]
    
    df3 = product_data.select([F.when(col(c) == '', None).otherwise(col(c)).alias(c) for c in ['title']])
    
    df3 = df3.na.fill(value = 'unknown', subset=['title'])
    
    numUnknowns_unknownImputedTitle = df3.where(df3.title == 'unknown').count()





    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': total_count,
        'mean_meanImputedPrice': meanOf_meanImputedPrice,
        'variance_meanImputedPrice': varOf_meanImputedPrice,
        'numNulls_meanImputedPrice': numNullsOf_meanImputedPrice,
        'mean_medianImputedPrice': meanOf_medianImputedPrice,
        'variance_medianImputedPrice': varOf_medianImputedPrice,
        'numNulls_medianImputedPrice': numNullsOf_medianImputedPrice,
        'numUnknowns_unknownImputedTitle': numUnknowns_unknownImputedTitle
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    product_processed_data = product_processed_data.withColumn(title_column, F.lower(col(title_column)))
    
    product_processed_data_output = product_processed_data.withColumn(titleArray_column, 
                                                                      F.split(col(title_column), ' '))
    
    word_2_vec = M.feature.Word2Vec(vectorSize = 16, minCount = 100, numPartitions = 4,
                                    seed = 102, inputCol = titleArray_column, outputCol = titleVector_column)
    
    
    
    
    model = word_2_vec.fit(product_processed_data_output)
    
    model.setInputCol(titleArray_column)
    
    size_vocabulary = model.getVectors().count()




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': [(None, None), ],
        'word_1_synonyms': [(None, None), ],
        'word_2_synonyms': [(None, None), ]
    }
    # Modify res:
    
    res['count_total'] = product_processed_data_output.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------

    ppd = product_processed_data

    stringIndexer = M.feature.StringIndexer(inputCol = 'category', outputCol = 'categoryIndex',
                                            stringOrderType = 'frequencyDesc')
    
    stringIndexerModel = stringIndexer.fit(ppd)
    
    df = stringIndexerModel.transform(ppd)
    
    
    
    OHE = M.feature.OneHotEncoder(inputCol = 'categoryIndex', outputCol = 'categoryOneHot', dropLast = False)
    
    OHEmodel = OHE.fit(df)
    
    
    
    
    
    intermediate_df = OHEmodel.transform(df)
    
    ct = intermediate_df.count()
    
    meanSummerizer = M.stat.Summarizer.mean
    
    meanVectorOfcatOH = intermediate_df.select(meanSummerizer(intermediate_df.categoryOneHot)).collect()[0][0]
    
    pca = M.feature.PCA(k=15, inputCol = categoryOneHot_column, outputCol = categoryPCA_column)
    
    PCAmodel = pca.fit(intermediate_df)
    
    final_df = PCAmodel.transform(intermediate_df)
    
    meanVectorOfcatPCA = final_df.select(meanSummerizer(final_df.categoryPCA)).collect()[0][0]




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': ct,
        'meanVector_categoryOneHot': meanVectorOfcatOH,
        'meanVector_categoryPCA': meanVectorOfcatPCA
    }
    # Modify res:




    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    decisionTree = M.regression.DecisionTreeRegressor(maxDepth = 5, featuresCol = 'features', 
                                                       labelCol = 'overall')
    dtModel = decisionTree.fit(train_data)
    
    dtPred = dtModel.transform(test_data)
    
    #class pyspark.ml.evaluation.RegressionEvaluator(*, predictionCol: str = 'prediction', labelCol: str = 'label',
    #metricName: RegressionEvaluatorMetricType = 'rmse',
    
    regrEvaluator = RegressionEvaluator(predictionCol = 'prediction', labelCol = 'overall')
    
    rmse = regrEvaluator.evaluate(dtPred, {regrEvaluator.metricName: 'rmse'})
    
    
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': rmse
    }
    # Modify res:


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    
    train_split, test_split = train_data.randomSplit([.75, .25])
    
    regrEvaluator = RegressionEvaluator(predictionCol = 'prediction', labelCol = 'overall')
    
    RMSEs = {}
    
    max_depths = [5, 7, 9, 12]
    
    for depth in max_depths:
        
        decisionTree = M.regression.DecisionTreeRegressor(maxDepth = depth, featuresCol = 'features', 
                                                       labelCol = 'overall')
        
        dtModel = decisionTree.fit(train_split)
        
        
        dtPred = dtModel.transform(test_split)
        
        rmse = regrEvaluator.evaluate(dtPred, {regrEvaluator.metricName: 'rmse'})
        
        
        RMSEs[depth] = rmse
        
    
        
        
    RMSEs_sorted = dict(sorted(RMSEs.items(), key = lambda score: score[1]))
    
    best_depth = list(RMSEs_sorted.keys())[0]
    
    bestDT = M.regression.DecisionTreeRegressor(maxDepth = best_depth, featuresCol = 'features', 
                                                       labelCol = 'overall')
    
    bestDTModel = bestDT.fit(train_data)
    
    bestDTPred = bestDTModel.transform(test_data)
    
    testRMSE = regrEvaluator.evaluate(bestDTPred, {regrEvaluator.metricName: 'rmse'})
    
    
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': testRMSE,
        'valid_rmse_depth_5': RMSEs[5],
        'valid_rmse_depth_7': RMSEs[7],
        'valid_rmse_depth_9': RMSEs[9],
        'valid_rmse_depth_12': RMSEs[12],
    }
    # Modify res:


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------

