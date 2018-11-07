import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import utils._


/**
  * Synthetic Minority Oversampling Technique (SMOTE) implemented in Scala and Spark
  *
  * @author Maria Jose Basgall - @mjbasgall
  */


object driver {

  def main(args: Array[String]) {

    if (args.length < 9) {
      System.err.println("Wrong number of parameters\n\t")
      System.exit(1)
    }


    // Disabling "INFO" level logs (these lines must be before to create the SparkContext)
    Logger.getRootLogger.setLevel(Level.ERROR)

    //Getting parameters from spark-submit
    val parameters = args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(param, value) => param -> value
        case Array(param) => param -> ""
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    // Parameters
    // Data:
    val headerFile = parameters("headerFile")
    val inputFile = parameters("inputFile")
    val delimiter = parameters.getOrElse("delimiter", ", ")

    val outputPah = parameters("outputPah")


    // Algorithm:
    val seed = parameters.getOrElse("seed", "1286082570").toInt
    val overPercentage = parameters.getOrElse("overPercentage", "100").toInt // > 0
    if (overPercentage <= 0) {
      System.err.println("The oversampling percentage must be > 0.\nYour value is\t: " + overPercentage)
      System.exit(1)
    } /*else if (overPercentage == 0) {
      System.err.println("The selected oversampling percentage doesn't change the current classes.\nYour value is\t: " + overPercentage)
      System.exit(1)
    } else if ((overPercentage % 100) != 0) {
      System.err.println("`overPercentage` must be multiple of 100.\nYour value is\t: " + overPercentage)
      System.exit(1)
    }*/


    val k = parameters.getOrElse("K", "5").toInt
    val numPartitions = parameters.getOrElse("numPartitions", "20").toInt
    val numReducers = parameters.getOrElse("nReducers", "1").toInt
    val numIterations = parameters.getOrElse("numIterations", "1").toInt // `1` if the train dataset fits in memory
    val minClassName = parameters.getOrElse("minClassName", "positive")

    // Spark basic setup
    val conf = new SparkConf().setAppName("SMOTE-BD")
    val sc = new SparkContext(conf)


    // Parse the header file to an array of maps, which will be used to transform the data into LabeledPoints (Spark API).
    val typeConversion = KeelParser.getParserFromHeader(sc, headerFile)
    val classes = typeConversion.apply(typeConversion.length - 1)
    println("Classes=\t" + classes)

    // Run SMOTE
    import java.time.{Duration, Instant}
    val start = System.currentTimeMillis()
    val before = Instant.now

    smote.runSMOTE_BD(sc, inputFile, delimiter, k, numPartitions, numReducers, typeConversion, numIterations, outputPah, seed, classes, minClassName, overPercentage)

    val stop = System.currentTimeMillis()
    val after = Instant.now
    val delta = Duration.between(before, after).toMillis
    val deltaMin = Duration.between(before, after).toMinutes
    println("The algorithm SMOTE-BD has finished running.\nTotalRuntime: " + (stop - start) + " ms \nTime by Instant: " + delta + " ms\t (" + deltaMin + " min.) \n")
    sc.stop()
  }
}
