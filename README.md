# BD4H
This repository will capture the work done by the team for spesis prediction using MIMIC III database as part of GeorgiaTech course

## Dependencies <br>
* scalaVersion := "2.11.12" <br>
 

* libraryDependencies ++= Seq(<br>
  "org.apache.spark" %% "spark-core" % "2.3.0", <br>
  "org.apache.spark" %% "spark-sql" % "2.3.1",<br><br>
  "com.databricks" %% "spark-xml" % "0.4.1",<br>
  "com.databricks" %% "spark-csv" % "1.5.0",<br>
  "org.apache.hadoop" % "hadoop-hdfs" % "2.7.2",<br>
  "org.apache.spark" %% "spark-mllib" % "2.3.0"<br>
)

-----

## How to Run the Code <br>
* Execute Spark(main.scala)
  * executes code to idenfity the sepsis date for patients who are identified with angus criteria of sepsis
  * executes code to find the date of final record of patients with non sepsis
  * executes code to filter data to extract information of blood vitals (heart rate, manual BP, Oxygen saturation,blood Temperature
  * extracts 20 topics from the patient notes and saves distribution of those 20 topics for each patient
 
* Python
  * Merge data of sepsis and non sepsis with their max dates
  * Merge date of patients with max dates and their blood vitals information
  * Extract the mean,median,standard deviation,len(count) summary details for each patient for following periods 
    * LastDay -- Day before sepsis was identified
    * last 7 days (week)
    * last 30 days
    * history of patients data
  * code to run logistic regression, CNN, RNN models
    
