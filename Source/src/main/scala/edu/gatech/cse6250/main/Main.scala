package edu.gatech.cse6250.main

import java.text.SimpleDateFormat

import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }
import org.apache.spark.mllib.clustering.{ LDA, OnlineLDAOptimizer, LocalLDAModel }
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{ DenseMatrix, Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import scala.io.Source
import org.apache.spark.ml.feature.{ CountVectorizer, RegexTokenizer, StopWordsRemover }
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.{ Vector => MLVector }
import java.io.File

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.sql.functions.{col,when,min}

/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Yu Jing <yjing43@gatech.edu>,
 * @author Ming Liu <mliu302@gatech.edu>
 */
object Main {
  
    def merge(srcPath: String, destPath: String): Unit = {

    val hadoopConfig = new Configuration()
    val hdfs = FileSystem.get(hadoopConfig)

    FileUtil.copyMerge(hdfs, new Path(srcPath), hdfs, new Path(destPath), false, hadoopConfig, null)
    FileUtil.fullyDelete(new File(srcPath))

  }
  
  /**
  * Filter data to fecth the blood vitals information
  */
  def filterDataForDatesBeforeSepsisMaxDate(val spark:SparkSession):Unit={
    val chartevents = "data/CHARTEVENTS.csv"

    val savePath = "data/filterChartData"
    val outputFile = "data/filteredChartData.csv"

    val sqlContext = spark.sqlContext

    List(chartevents)
      .foreach(CSVHelper.loadCSVAsTable(spark,_))

    sqlContext.sql(
      """
        | SELECT *
        | from chartevents
        | where itemid in (211,220045,442,224167,228234,228232,226329)
      """.stripMargin
    ).write.csv(savePath)


    merge(savePath,outputFile)
  }
  
   /**
  * Identify Max Date for Non Sepsis patients
  */
  def nonSepsisMaxDate(val spark: SparkSession):Unit = {
    
      val labevents = "data/LABEVENTS.csv"
    val chartevents = "data/CHARTEVENTS.csv"
    val noteevents = "data/NOTEEVENTS.csv"

    val savePath = "data/labevents"
    val outputFile = "data/NoSepsisoutputWithSepTime.csv"

    val sqlContext = spark.sqlContext

    List(labevents,chartevents,noteevents)
      .foreach(CSVHelper.loadCSVAsTable(spark,_))

    val df = sqlContext.sql(
      """
        | SELECT subject_id, MAX(to_date(charttime)) as finalDate
        | from labevents
        | group by subject_id
      """.stripMargin
    )

      val df1 = sqlContext.sql(
        """
          | SELECT subject_id, MAX(to_date(charttime)) as finalDate
          | from chartevents
          | group by subject_id
        """.stripMargin
      )

    val df2 = sqlContext.sql(
      """
        | SELECT subject_id, MAX(to_date(charttime)) as finalDate
        | from noteevents
        | group by subject_id
      """.stripMargin
    )

      val outputdf = df.union(df1).union(df2)
    
    outputdf.groupBy(col("subject_id"))
      .agg(max(col("finalDate")))
      .write.csv(savePath)


    merge(savePath,outputFile)
    
  }
  
    /**
  * Date on which sepsis was first Identified
  */
  def Cse6250_SepsisIdentificationDate(val spark: SparkSession): Unit = {
   
        val admissions = "data/ADMISSIONS_Demo.csv"
    val procedures = "data/PROCEDURES_ICD_Demo.csv"
    val diagnoses = "data/DIAGNOSES_ICD_Demo.csv"
    val icustays = "data/ICUSTAYS_Demo.csv"

    val savePath = "data/infectionFiles"
    val outputFile = "data/outputWithSepTime.csv"

    val sqlContext = spark.sqlContext

    List(admissions,procedures,diagnoses,icustays)
      .foreach(CSVHelper.loadCSVAsTable(spark,_))

    val infectionGroup = sqlContext.sql(
      """
        |
        |SELECT subject_id,hadm_id,infection,min(C.INTIME) as infectionTime from
        |(
        | SELECT A.subject_id,A.hadm_id, to_date(B.INTIME) as INTIME,
        |	CASE
        |		WHEN substring(icd9_code,1,3) IN ('001','002','003','004','005','008',
        |			   '009','010','011','012','013','014','015','016','017','018',
        |			   '020','021','022','023','024','025','026','027','030','031',
        |			   '032','033','034','035','036','037','038','039','040','041',
        |			   '090','091','092','093','094','095','096','097','098','100',
        |			   '101','102','103','104','110','111','112','114','115','116',
        |			   '117','118','320','322','324','325','420','421','451','461',
        |			   '462','463','464','465','481','482','485','486','494','510',
        |			   '513','540','541','542','566','567','590','597','601','614',
        |			   '615','616','681','682','683','686','730') THEN 1
        |		WHEN substring(icd9_code,1,4) IN ('5695','5720','5721','5750','5990','7110',
        |				'7907','9966','9985','9993') THEN 1
        |		WHEN substring(icd9_code,1,5) IN ('49121','56201','56203','56211','56213',
        |				'56983') THEN 1
        |		ELSE 0 END AS infection
        |	FROM diagnoses_icd A, icustays B
        | where A.hadm_id = B.hadm_id and A.subject_id = B.subject_id
        | ) C
        | where C.infection = 1
        | group by C.subject_id,C.hadm_id,C.infection
      """.stripMargin
    ).cache()


    val expicitSepsis = sqlContext.sql(
      """
        |SELECT subject_id,hadm_id, min(C.INTIME) as sepTime from
        |(
        |SELECT A.subject_id,A.hadm_id, to_date(B.INTIME) as INTIME,
        |		-- Explicit diagnosis of severe sepsis or septic shock
        |		CASE
        |		WHEN substring(icd9_code,1,5) IN ('99592','78552')  THEN 1
        |		ELSE 0 END AS explicit_sepsis
        |	FROM diagnoses_icd A, icustays B
        | where A.hadm_id = B.hadm_id and A.subject_id = B.subject_id
        | ) C
        | where C.explicit_sepsis = 1
        | group by C.subject_id,C.hadm_id, C.explicit_sepsis
      """.stripMargin
    )

    val organDysfunction = sqlContext.sql(
      """
        |SELECT subject_id,hadm_id,organ_dysfunction, min(C.INTIME) as organDysFunctionTime from
        |(
        |SELECT A.subject_id,A.hadm_id, to_date(B.INTIME) as INTIME,
        |		CASE
        |		-- Acute Organ Dysfunction Diagnosis Codes
        |		WHEN substring(icd9_code,1,3) IN ('458','293','570','584') THEN 1
        |		WHEN substring(icd9_code,1,4) IN ('7855','3483','3481',
        |				'2874','2875','2869','2866','5734')  THEN 1
        |		ELSE 0 END AS organ_dysfunction
        |	FROM diagnoses_icd A, icustays B
        | where A.hadm_id = B.hadm_id and A.subject_id = B.subject_id
        | ) C
        | where C.organ_dysfunction = 1
        | group by C.subject_id,C.hadm_id, C.organ_dysfunction
      """.stripMargin
    )

    val organProcGroup = sqlContext.sql(
      """
        |SELECT subject_id,hadm_id, mech_vent, min(C.INTIME) as organProcGroupTime from
        |(
        |SELECT A.subject_id,A.hadm_id, to_date(B.INTIME) as INTIME,
        |		CASE
        |		WHEN substring(icd9_code,1,4) IN ('9670','9671','9672') THEN 1
        |		ELSE 0 END AS mech_vent
        |	FROM procedures_icd A, icustays B
        | where A.hadm_id = B.hadm_id and A.subject_id = B.subject_id
        | ) C
        | where mech_vent = 1
        | group by C.subject_id ,C.hadm_id, mech_vent
      """.stripMargin
    )


    val infectOrganDysSepsis = infectionGroup.join(organDysfunction,Seq("subject_id","hadm_id"))
      .withColumn("sepTime",
        when(col("infectionTime") > col("organDysFunctionTime"),col("infectionTime"))
      .otherwise(col("organDysFunctionTime")))
      .select(col("subject_id"),col("hadm_id"),col("sepTime"))
//      .write.csv(savePath)

//    infectOrganDysSepsis.printSchema()
//    df1.printSchema()


    val infectMechEventSepsis = infectionGroup.join(organProcGroup,Seq("subject_id","hadm_id"))
      .withColumn("sepTime",
        when(col("infectionTime") > col("organProcGroupTime"),col("infectionTime"))
          .otherwise(col("organProcGroupTime")))
      .select(col("subject_id"),col("hadm_id"),col("sepTime"))
//      .write.csv(savePath)

    expicitSepsis.union(infectOrganDysSepsis).union(infectMechEventSepsis)
      .groupBy(col("subject_id"))
      .agg(min(col("sepTime")))
      .write.csv(savePath)


    merge(savePath,outputFile)
    
    
  } 
  
  def main(args: Array[String]) {
    import org.apache.log4j.{ Level, Logger }

    val numTopics: Int = 20
    val maxIterations: Int = 100
    val vocabSize: Int = 10000

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark
    val sc = spark.sparkContext
    val sqlContext = spark.sqlContext
    
    Cse6250_SepsisIdentificationDate(spark)
    nonSepsisMaxDate(spark)
    filterDataForDatesBeforeSepsisMaxDate(spark)
    
    import sqlContext.implicits._

    val rawNotes = CSVHelper.loadCSVAsTable(spark, "data/NOTEEVENTS_Demo.csv")

    val rawTextRDD = rawNotes.map(r => r.getAs[String](10)).rdd
    println(rawTextRDD.count)
    //rawTextRDD.take(10).foreach(println)

    val docRdd = rawTextRDD.zipWithIndex

    // docRdd.take(10).foreach(println)
    val CustomSchema = StructType(Array(StructField("text", StringType, true), StructField("docId", IntegerType, true)))

    val docDF = sqlContext.createDataFrame(docRdd)

    val newNames = Seq("text", "docId")
    val docDF3 = docDF.toDF(newNames: _*)

    // Split each document into words
    val tokens = new RegexTokenizer()
      .setGaps(false)
      .setPattern("\\p{L}+")
      .setInputCol("text")
      .setOutputCol("words")
      .transform(docDF3)

    // Filter out stopwords
    val stopwords: Array[String] = Array("a", "an", "the", "and", "but", "if", "please", "\n", "she", "him", "her", "admission", "to", "on", "with", "for", "in", "was", "of", "it", "from", "is", "had", "have", "at", "no", "he", "this", "at", "there", "am", "patient", "hospital", "are", "o", "name", "s", "doctor", "as", "not", "mg", "ml", "or", "home", "left", "right", "today", "previous", "dl", "hr", "reason", "w", "be", "pm", "c", "pt", "htc", "during", "although", "remain", "small", "within", "gj", "po", "t", "ct", "up", "his", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "imagine", "maybe", "plan", "action", "assesment", "care", "via", "cc", "ef", "rsc", "own", "real", "hoping", "mm", "onset", "date", "by", "has", "which", "kg", "due", "seen", "other", "assessed", "day", "last", "now", "given", "per", "hx", "bp", "bed", "bedtime", "since", "bs", "tf", "osh", "normal", "assessment", "when", "hct", "hours", "will", "needed", "ni", "remains", "meq", "been", "started", "iv", "may", "team", "that")
    val filteredTokens = new StopWordsRemover()
      .setStopWords(stopwords)
      .setCaseSensitive(false)
      .setInputCol("words")
      .setOutputCol("filtered")
      .transform(tokens)

    //filteredTokens.collect().foreach(println)

    // Limit to top `vocabSize` most common words and convert to word count vector features
    val cvModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setVocabSize(vocabSize)
      .fit(filteredTokens)

    //val countV = cvModel.transform(filteredTokens)
    //print("class of CountV " + countV.getClass.toString())
    //print(countV)

    val countVectors = cvModel
      .transform(filteredTokens)
      .select("docId", "features")
      .rdd.map { case Row(docId: Long, features: MLVector) => (docId.toLong, Vectors.fromML(features)) }

    //countVectors.collect().foreach(println)

    //val countVectors = countVectors1.rdd

    /**
     * Configure and run LDA
     */

    //val corpusSize = countVectors.count()

    val mbf = {
      // add (1.0 / actualCorpusSize) to MiniBatchFraction be more robust on tiny datasets.
      val corpusSize = countVectors.count()
      2.0 / maxIterations + 1.0 / corpusSize
    }

    val lda = new LDA()
      .setOptimizer(new OnlineLDAOptimizer().setMiniBatchFraction(math.min(1.0, mbf)))
      .setK(numTopics)
      .setMaxIterations(2)
      .setDocConcentration(-1) // use default symmetric document-topic prior
      .setTopicConcentration(-1) // use default symmetric topic-word prior

    val startTime = System.nanoTime()
    val ldaModel = lda.run(countVectors)
    val elapsed = (System.nanoTime() - startTime) / 1e9

    /**
     * Print results.
     */
    // Print training time
    println(s"Finished training LDA model.  Summary:")
    println(s"Training time (sec)\t$elapsed")
    println(s"==========")

    // Print the topics, showing the top-weighted terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    val vocabArray = cvModel.vocabulary
    val topics = topicIndices.map {
      case (terms, termWeights) =>
        terms.map(vocabArray(_)).zip(termWeights)
    }
    /**
     * println(s"$numTopics topics:")
     * topics.zipWithIndex.foreach {
     * case (topic, i) =>
     * println(s"TOPIC $i")
     * topic.foreach { case (term, weight) => println(s"$term\t$weight") }
     * println(s"==========")
     * }
     */
    var localLDAModel = ldaModel.asInstanceOf[LocalLDAModel];
    val topicDistributionsOverDocument = localLDAModel.topicDistributions(countVectors)
    //topicDistributionsOverDocument.collect().foreach(println)
  }
}
