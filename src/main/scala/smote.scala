import java.time.{Duration, Instant}

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.{LabeledPoint, MinMaxScaler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import utils._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Distributed SMOTE
  *
  * Maria José Basgall
  */
object smote {

  type InstanceKey = Long
  type InstanceKeyLp = (InstanceKey, LabeledPoint)
  type KNeighborsKeys = Array[InstanceKey]
  type InstanceAndItsNeighbors = (InstanceKey, KNeighborsKeys)


  //To use with ``sample`` operation
  val with_replacement = true
  val without_replacement = true
  //To use with ``mapPartitions``
  val Preserves = true


  val DontShuffle = false

  /**
    *
    * @param sc            SparkContext
    * @param inPath        The complete input file path
    * @param delimiter     The character which delimits each feature
    * @param k             How many neighbours to get
    * @param numPartitions The number of maps to use
    * @param numReducers   The number of reducers to use in kNN algorithm
    * @param bcTypeConv    The parsed header file
    * @param numIterations `1` if train data fits in memory
    * @param outPath       output path
    * @param seed          to take samples
    */
  def runSMOTE_BD(sc: SparkContext,
                  inPath: String,
                  delimiter: String,
                  k: Int,
                  numPartitions: Int,
                  numReducers: Int,
                  bcTypeConv: Array[Map[String, Double]],
                  numIterations: Int,
                  outPath: String,
                  seed: Int,
                  classes: Map[String, Double],
                  minClassName: String,
                  overPerc: Int): Unit = {

    val minClassNumber = classes.apply(minClassName)
    val majClassNumber = classes.find(_._1 != minClassName).get._2
    val majClassName = classes.find(_._1 != minClassName).get._1

    println("The minority class name is:\t" + minClassName + "\t and the number given for Keel is:\t" + minClassNumber)
    println("The majority class name is:\t" + majClassName + "\t and the number given for Keel is:\t" + majClassNumber)

    /*  Get each point of the input file as a LabeledPoint: [double, Vector[double]].
        For instance: (0.0,[0.0,0.56,0.45,0.185,1.07,0.3805,0.175,0.41]) */

    // `numPartitions` is a suggested minimum number of partitions for the resulting RDD
    val allData = sc.textFile(inPath: String, numPartitions).filter(line => !line.isEmpty && line.split(",").length == bcTypeConv.length)
      .map(line => KeelParser.parseLabeledPoint(bcTypeConv, line))
    val numAll = allData.count()
    println("INFO FOR allData " + printRDDInfo(allData))


    /*
      MinMaxScaler - Data normalization
     */
    val sqlContext = SparkSession
      .builder()
      .appName("SMOTE-BD")
      .getOrCreate()


    // The following import doesn't work externally because the implicits object is defined inside the SQLContext class
    import sqlContext.implicits._

    // Create a DataFrame from the RDD[LabeledPoint], with the following two columns:
    //|labels|            features|
    //+------+--------------------+
    //|   1.0|[6.0,7.0,42.0,1.1...|
    val allDF = allData.map(e => (e.label, e.features)).toDF("labels", "features") //.cache()
    //allDF.show(5,false)

    // MaxMin Scaler with the min(0) and max(1)
    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("featuresScaled")
      .setMax(1)
    // .setMin(0)

    // Scaling and drop the `features` column
    /*  +------+--------------------+--------------------+
        |labels|            features|      featuresScaled|
        +------+--------------------+--------------------+
        |   1.0|[6.0,7.0,42.0,1.1...|[1.0,0.2142857142...|*/
    val fittedDF = scaler.fit(allDF)
    val vector_original_min = fittedDF.originalMin
    val vector_original_max = fittedDF.originalMax

    //println("\nvector_original_min\t" + vector_original_min)
    //println("vector_original_max\t" + vector_original_max)

    // Apply transforming over the minority data only (but using the whole statistic of allDF)
    val normalizedDF = fittedDF.transform(allDF.filter($"labels" === minClassNumber)).drop("features").withColumn("idx", monotonically_increasing_id())

    // Convert from the DataFrame to RDD
    /* After normalization, apply repartition to put data across the `numPartitions` partitions (because in the reading file part the number of partitions is taken only as suggestion) */
    val posData = normalizedDF.rdd.map {
      row =>
        (row.getAs[Long]("idx"),
          LabeledPoint(
            row.getAs[Double]("labels"),
            row.getAs[Vector]("featuresScaled")
          )
        )
    }.repartition(numPartitions)

    posData.cache()


    val numPos = posData.count()
    val numNeg = numAll - numPos
    println("\nNumber of MINORITY class instances (" + minClassName + ") :\t" + numPos)
    println("Number of MAJORITY class instances (" + majClassName + "):\t" + numNeg)

    if (numPos <= k) {
      System.err.println("Positive instances must be greater than 'k' value")
      System.exit(1)
    }
    println("----\nThe selected `k` value is\t" + k)
    println("The empirical rule (Math.sqrt(numPos)) for `k` calculation gets\t" + Math.sqrt(numPos))
    println("----\n")

    /*
    The minority class will be modified an `overPerc` % comparing with the num of elements of the majority class.

    Thus, for instance, if numNeg = 500 and numPos = 50,
    If overPerc = 100   (1.0x) =>  finalQtyPos = 500 and numSyn = 450 new instances. Both the same quantity of intances.
    If overPerc = 50    (0.5x) =>  finalQtyPos = 250 and numSyn = 200 new instances. MinClass will have half number of elem than MajClass.
    If overPerc = 150   (1.5x) =>  finalQtyPos = 750 and numSyn = 700 new instances.
    If overPerc = 200   (2.0x) =>  finalQtyPos = 1000 and numSyn = 950 new instances. MinClass will have twice as many elements than MajClass.
     */
    val finalQtyPos = math.ceil(numNeg * overPerc / 100).toInt //round up
    val numSyn = finalQtyPos - numPos
    val negPerc = numNeg * 100 / numAll
    val posPerc = numPos * 100 / numAll
    println("Current classes distribution\t NEG:\t" + negPerc + " %\t- POS:\t " + posPerc + " %")
    println("Desired oversampling percentage (to apply on the minority class): \t" + overPerc + " %.")
    println("The NEW quantity of MINORITY class instances (" + minClassName + ") will be:\t" + finalQtyPos)
    var creationFactor = math.ceil(numSyn / numPos).toInt //round up
    if ((creationFactor % numPos) != 0) creationFactor = creationFactor + 1
    println("Each minority instance has to create \t" + creationFactor + " artificial points")
    println("IT HAS TO BE CREATED:\t" + numSyn + " instances\tto get the desired oversampling percentage.")

    //println("INFO FOR posData\t" + printRDDInfo(posData))


    //printRDDInfo(posData)
    //println("posData\t==================+++++++++++++++++++++++++++++++++++++++++++++++==============================")
    //printRddContent(posData, posData.count().toInt)
    //posData.foreach(println)

    /*
    kNN stage

    Calculates kNN for each instance using a slightly modified version based on the Jesús Maillo implementation.

    @Notes: `numIterations = 1` when train data fits in memory

    @return As result of kNN, each neighbour will have its Key and the keys of its neighbours: (Key,Array[Keys]). This returns an RDD with only one partition.

     */
    val beforeKnn = Instant.now
    val globalkNN = kNN_IS.setup(posData, posData, k, numReducers, numIterations).calculatekNeighbours() //.cache()

    val afterKnn = Instant.now
    val deltaKnn = Duration.between(beforeKnn, afterKnn).toMillis
    println("Time for the kNN stage\t" + deltaKnn + " ms\n")

    //printRDDInfo(globalkNN)
    //println("Number of globalkNN " + globalkNN.count())
    //println("printing globalkNN -------------------------------------------------------------------------")
    //printRddContent(globalkNN, 4)


    /* Link the neighbours with the dataWithKey (LabeledPoint), joining them using their Keys.
     And for each Key, produce a pair of (LabeledPoint, Array[String]) as result
     */
    val joinedData = globalkNN.join(posData).map { case (key, (neighbours_keys, lp)) => (key, neighbours_keys, lp) }

    /* Convert the RDD[T] to RDD[Array[T]] */
    val trainDataArray = posData.mapPartitions(x => Iterator(x.toArray), preservesPartitioning = Preserves).cache()

    /*
    SMOTE stage
     */
    val beforeSMT = Instant.now
    val synData = applySMOTEBD(sc,
      trainDataArray,
      joinedData,
      delimiter,
      creationFactor,
      k,
      minClassNumber,
      minClassName,
      majClassName,
      seed) //.cache()

    val afterSMT = Instant.now
    val deltaSMT = Duration.between(beforeSMT, afterSMT).toMillis
    println("Time for the SMT stage\t" + deltaSMT + "\tms\n")

    println("Number of created instances\t" + synData.count())


    /*
    Cutting synData
     */
    val beforeTake = Instant.now
    val cutSynData = sc.parallelize(synData.take(numSyn.toInt)) //take only the exact number and make it RDD again
    val afterTake = Instant.now
    val deltaTake = Duration.between(beforeTake, afterTake).toMillis
    println("Time for the cutting\t" + deltaTake + "\tms\n")


    /**
      * Data de-normalization
      */
    val beforeDenorm = Instant.now
    // total number of columns (features + class)
    val numCols = bcTypeConv.length
    val selectCols = (0 until (numCols - 1)).map(i => $"arr" (i).as(s"col_$i"))

    /* convert the RDD[String] to RDD[Array[String]], and take off the labels of the converted DF */
    val synDataNoLabelsDF = cutSynData.map(e => e.split(",")).toDF("arr").select(selectCols: _*)

    val updateFunction = (columnValue: Column, minValue: Double, maxValue: Double) =>
      (columnValue * (lit(maxValue) - lit(minValue))) + lit(minValue)

    val updateColumns = (df: DataFrame, minVector: Vector, maxVector: Vector, updateFunction: (Column, Double, Double) => Column) => {
      val columns = df.columns
      minVector.toArray.zipWithIndex.map {
        case (_, index) => updateFunction(col(columns(index)), minVector(index), maxVector(index)).as(columns(index))
      }
    }

    // Applying de-normalization and adding `minClassName` class column
    val denormdDF = synDataNoLabelsDF.select(
      updateColumns(synDataNoLabelsDF, vector_original_min, vector_original_max, updateFunction): _*
    ).withColumn("col_" + numCols, lit(minClassName))


    // Convert Dataframe to RDD[String]
    val synRDD = denormdDF.rdd.map(_.mkString(", "))

    val afterDenorm = Instant.now
    val deltaDenorm = Duration.between(beforeDenorm, afterDenorm).toMillis
    println("Time for the denormalization stage\t" + deltaDenorm + "\tms\n")

    synRDD.cache()
    val numSynRDD = synRDD.count()

    /* Convert RDD[LabeledPoint] to RDD[String] */
    val originalData = allData.map(x => x.features.toArray.mkString(delimiter) + delimiter + (if (x.label == minClassNumber) minClassName else majClassName)).cache()


    /* Save results */
    val fs = FileSystem.get(sc.hadoopConfiguration)
    if (fs.exists(new Path(outPath))) {
      println("Output directory already exists. Deleting...")
      fs.delete(new Path(outPath), true)
    }

    val outputData = synRDD.union(originalData)
    outputData.saveAsTextFile(outPath, classOf[org.apache.hadoop.io.compress.GzipCodec]) // save results in compressed part-... files
    println("Results have been saved on\t" + outPath)

    val posPercFinal = ((numPos + numSynRDD) * 100) / numNeg

    println("FINAL classes distribution\t NEG:\t100 %\t- POS:\t " + posPercFinal + " %")
    println("FINAL classes quantities\t NEG:\t" + numNeg + " \t- POS:\t " + (numPos + numSynRDD) + " ")
  }


  /**
    *
    * @param sc                 SparkContext
    * @param trainData          positive data points
    * @param dataWithNeighbours neighbourdhood
    * @param delimiter          features delimiter
    * @return synthetic dataset
    */
  private def applySMOTEBD(sc: SparkContext,
                           trainData: RDD[Array[InstanceKeyLp]],
                           dataWithNeighbours: RDD[(InstanceKey, KNeighborsKeys, LabeledPoint)],
                           delimiter: String,
                           creationFactor: Int,
                           k: Int,
                           minClassNumber: Double,
                           minClass: String,
                           majClass: String,
                           seed: Int
                          ): RDD[String] = {


    val neighboursBroadcast = sc.broadcast(dataWithNeighbours.collect())

    var synDataTmp: RDD[String] = null

    /* Send the data belong to each partition to create the synthetic data */
    synDataTmp = trainData.mapPartitionsWithIndex(createSyntheticData(_, _, neighboursBroadcast, delimiter, creationFactor, k, minClassNumber, minClass, majClass, seed), preservesPartitioning = Preserves) //.cache()

    synDataTmp
  }


  /**
    * Create artificial instances
    *
    * @param partitionIndex  index of the partition
    * @param partitionDataIt all data belonging to `partitionIndex` partition
    * @param neighboursBr    a broadcast which contains the neighbourhood
    * @param delimiter       features delimiter
    * @param creationFactor  number of instances to be created for each positive example
    * @param k               neighbours number
    * @return artificial instances for the current partition
    */
  private def createSyntheticData(partitionIndex: Long,
                                  partitionDataIt: Iterator[Array[InstanceKeyLp]],
                                  neighboursBr: Broadcast[Array[(Long, Array[Long], LabeledPoint)]],
                                  delimiter: String,
                                  creationFactor: Int,
                                  k: Int,
                                  minClassNumber: Double,
                                  minClass: String,
                                  majClass: String,
                                  seed: Int
                                 ): Iterator[String] = {

    var artificialData = ArrayBuffer[String]()

    if (partitionDataIt.nonEmpty) {
      val partitionData = partitionDataIt.next
      //println("Part #\t" + partitionIndex + "\t\tpartitionData Elemnts #\t" + partitionData.length)

      val global_neighbours = neighboursBr.value
      val globalNeighMapByKey = global_neighbours.map { case (key, neighkeys, lp) => (key, (lp, neighkeys)) }.toMap


      val label = if (global_neighbours.last._3.label == minClassNumber) minClass else majClass
      var neighLp: Vector = null
      var newInstance: String = ""
      var neighKey: InstanceKey = 0
      val rand = new scala.util.Random(seed)
      var neighList: KNeighborsKeys = null

      for {
        e <- partitionData
        num_created <- 0 until creationFactor
        _ = {
          neighList = globalNeighMapByKey(e._1)._2

          neighKey = neighList(rand.nextInt(k)) //key of a random neighbour
          neighLp = globalNeighMapByKey(neighKey)._1.features

          // instances interpolation
          newInstance = interpolation(e._2.features, neighLp).toString

          // cut the brackets
          artificialData.+=(newInstance.substring(1, newInstance.length() - 1) + delimiter + label)

        }
      } yield ()
    }

    artificialData.iterator
  }


  /**
    * Create an artificial instance
    *
    * @param sf sampleFeatures (labeledPoint Vector)
    * @param nf neighbourFeatures (labeledPoint Vector)
    * @return the new features vector for the artificial instance
    */
  private def interpolation(sf: Vector, nf: Vector): Vector = {
    val size = sf.size
    val rand = new Random().nextDouble()
    val result = new Array[Double](size)
    var difference = 0.0

    for (i <- 0 until size) {
      difference = (nf(i) - sf(i)) * rand
      result(i) = sf(i) + difference
    }
    Vectors.dense(result)
  }


  /**
    *
    * @param anRdd to print its information
    * @tparam A represents any type of data
    * @return
    */
  def printRDDInfo[A](anRdd: RDD[A]): String = {
    // Get info of each partition
    //`.glom()` applies coalesce and return as an array
    val info = anRdd.glom().map(_.length).collect()
    val avg = info.sum / info.length

    if (info.min <= 5) {
      System.err.println("It is NOT POSSIBLE to run knn with only one sample in SOME PARTITION OF the training set due to the `sample` function will repeated all points! Please, try again with this number of partition for this dataset:\t" + (info.length - 1) + "\n\t")
      System.exit(1)
    }


    val str = "Min:\t" + info.min + " - Max:\t" + info.max + " - avg:\t " + avg + " - numParts:\t" + info.length
    str
  }

  /**
    *
    * @param anRdd to print its content
    * @param n     the number of elements to show
    * @tparam A represents any type of data
    */
  def printRddContent[A](anRdd: RDD[A], n: Int): Unit = {
    anRdd match {
      case r1: RDD[InstanceAndItsNeighbors] => r1.asInstanceOf[RDD[InstanceAndItsNeighbors]].map { case (a, arr) => (a, arr.toList) }.take(n).foreach(println)
      case r2: RDD[(InstanceKey, LabeledPoint)] => r2.asInstanceOf[RDD[(InstanceKey, LabeledPoint)]].map { case (a, arr) => (a, arr.toString()) }.take(n).foreach(println)
      case _ => anRdd.foreach(println)
    }
    println()
  }
}
