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

/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Yu Jing <yjing43@gatech.edu>,
 * @author Ming Liu <mliu302@gatech.edu>
 */
object Main {
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
