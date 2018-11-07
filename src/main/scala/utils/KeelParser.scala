package utils

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors

import scala.collection.mutable.ArrayBuffer


/**
  * Gets the information from the header in order to normalize or parser to LabeledPoint or Array[Double] a data set.
  *
  * @author Jesus Maillo
  * @note Some minor modifications were made by Maria Jose Basgall (@mjbasgall)
  */

object KeelParser {

  /**
    * Get the information necessary for parser with function parserToDouble
    *
    * @param sc   The SparkContext
    * @param file path of the header
    * @author jesus
    */
  def getParserFromHeader(sc: SparkContext, file: String): Array[Map[String, Double]] = {
    //Reading header. Each element is a line
    val header = sc.textFile(file)
    var linesHeader = header.collect() //toArray()

    //Calculate number of features + 1 for the class
    var numFeatures = 0
    //for (i <- 0 until linesHeader.length) {
    for (i <- linesHeader.indices) {
      if (linesHeader(i).toUpperCase().contains("@INPUTS")) {
        numFeatures = linesHeader(i).length - linesHeader(i).replaceAllLiterally(",", "").length + 2
      } //end if
    } //end for

    //Calculate transformation to normalize and erase categorical features
    val conv = new Array[Map[String, Double]](numFeatures)
    for (i <- 0 until numFeatures) conv(i) = Map()

    var auxParserClasses = 0.0
    var auxNumFeature = 0
    // for (i <- 0 until linesHeader.length) {
    for (i <- linesHeader.indices) {
      if (linesHeader(i).toUpperCase().contains("@ATTRIBUTE CLASS")) {
        val labelsClasses = getLabels(linesHeader(i)) //Array of String with the labels of the objetive variable
        for (key <- labelsClasses) { //Calculate map for parser label classes
          conv(numFeatures - 1) += (key -> auxParserClasses)
          auxParserClasses = auxParserClasses + 1
        }
      } else if (linesHeader(i).toUpperCase().contains("[")) { //Real or integer feature
        val range = getRange(linesHeader(i)) //Min and max of the feature
        conv(auxNumFeature) += ("min" -> range(0), "max" -> range(1)) //Do the parser for this feature
        auxNumFeature = auxNumFeature + 1 //Increase for the next feature
      } else if (linesHeader(i).toUpperCase().contains("{") && !(linesHeader(i).toUpperCase().contains("@ATTRIBUTE CLASS"))) {
        var auxParserCategories = 0.0
        val labelsClasses = getLabels(linesHeader(i)) //Array of String with the labels of the objetive variable
        for (key <- labelsClasses) { //Calculate map for parser label categorical attributes
          conv(auxNumFeature) += (key -> auxParserCategories)
          auxParserCategories = auxParserCategories + 1
        }
        auxNumFeature = auxNumFeature + 1
      }
    } //end for

    conv
  }


  /**
    * Get the labels of a feature or the main class as Array[String]
    *
    * @param str string to parser
    * @author jesus
    */
  def getLabels(str: String): Array[String] =
    str.substring(str.indexOf("{") + 1, str.indexOf("}")).replaceAll(" ", "").split(",")


  /**
    * Get the min and max of a feature as a Array[Double]
    *
    * @param str string to parser
    * @author jesus
    */
  def getRange(str: String): Array[Double] = {
    val aux = str.substring(str.indexOf("[") + 1, str.indexOf("]")).replaceAll(" ", "").split(",")
    val result = new Array[Double](2)
    result(0) = aux(0).toDouble
    result(1) = aux(1).toDouble
    result
  }


  /**
    * Parser a line to a Array[Double]
    *
    * @param conv Array[Map] with the information to parser
    * @param line The string to be parsed
    * @author jesus
    */
  def parserToDouble(conv: Array[Map[String, Double]], line: String): Array[Double] = {
    val size = conv.length
    val result: Array[Double] = new Array[Double](size)

    //Change the line to Array[String]
    val auxArray = line.split(",")

    //Iterate over the array parsing to double each element with the knwolegde of the header
    for (i <- 0 until size) {
      if (conv(i).contains("min") && conv(i).contains("max") && (conv(i).size == 2)) { //If dictionary have like key (only) min and max is real or integer, else, categorical
        //result(i) = (auxArray(i).toDouble - conv(i).get("min").get) / (conv(i).get("max").get - conv(i).get("min").get)
        result(i) = (auxArray(i).toDouble - conv(i)("min")) / (conv(i)("max") - conv(i)("min"))
      } else {
        result(i) = conv(i)(auxArray(i))
      }
    }
    result
  }

  def parserCategoricalToDouble(conv: Array[Map[String, Double]], line: String): Array[Double] = {
    val size = conv.length
    val result: Array[Double] = new Array[Double](size)

    //Change the line to Array[String]
    val auxArray = line.split(",")

    //Iterate over the array parsing to double each element with the knwolegde of the header
    for (i <- 0 until size) {
      if (conv(i).contains("min") && conv(i).contains("max") && (conv(i).size == 2)) { //If dictionary have like key (only) min and max is real or integer, else, categorical
        result(i) = auxArray(i).toDouble
      } else {
        result(i) = conv(i)(auxArray(i).trim())
      }
    }
    result
  }

  def parserCategoricalToDoubleResultString(conv: Array[Map[String, Double]], line: String): String = {
    val size = conv.length
    val result: Array[String] = new Array[String](size)

    //Change the line to Array[String]
    val auxArray = line.split(",")

    //Iterate over the array parsing to double each element with the knwolegde of the header
    for (i <- 0 until size) {
      if (conv(i).contains("min") && conv(i).contains("max") && (conv(i).size == 2)) { //If dictionary have like key (only) min and max is real or integer, else, categorical
        result(i) = auxArray(i) //- conv(i).get("min").get) / (conv(i).get("max").get - conv(i).get("min").get)
      } else {
        result(i) = conv(i)(auxArray(i).trim()).toString
      }
    }
    result.mkString(",")
  }


  def getNumClassFromHeader(sc: SparkContext, file: String): Int = {
    var numClass = 0
    val header = sc.textFile(file)
    val linesHeader = header.collect() //toArray()

    for (i <- linesHeader.indices) {
      if (linesHeader(i).toUpperCase().contains("@ATTRIBUTE CLASS")) {
        numClass = linesHeader(i).length - linesHeader(i).replaceAllLiterally(",", "").length + 1
      } //end if
    } //end for

    numClass

  }

  def parseLabeledPoint(conv: Array[Map[String, Double]], str: String): LabeledPoint = {

    val tokens = str.split(",")
    require(tokens.length == conv.length)

    val arr = (conv, tokens).zipped.map { (c, elem) =>
      c.getOrElse(elem.trim, elem.trim.toDouble)
    }


    val features = arr.slice(0, arr.length - 1)
    val label = arr.last

    LabeledPoint(label, Vectors.dense(features))
  }

  def parseString(conv: Array[Map[String, Double]], str: String): String = {

    val tokens = str.split(",")
    require(tokens.length == conv.length)

    val arr = (conv, tokens).zipped.map { (c, elem) =>
      c.getOrElse(elem.trim, elem.trim.toDouble)
    }
    val result: Array[String] = arr.map(element => element.toString())
    result.mkString(",")
  }

  def getPositionAttributesNumeric(sc: SparkContext, file: String): ArrayBuffer[Int] = {
    val position = ArrayBuffer[Int]()
    val header = sc.textFile(file)
    var linesHeader = header.collect() //toArray()
    for (pos <- linesHeader.indices) {
      if (linesHeader(pos).toUpperCase().contains("[") || (linesHeader(pos).toUpperCase().contains("@ATTRIBUTE CLASS"))) {
        position += pos - 1
      }
    }
    position
  }

  def getAttributesNumeric(position: ArrayBuffer[Int], str: String): String = {
    val tokens = str.split(",")
    val result: Array[String] = new Array[String](position.length)
    for (i <- 0.to(position.length - 2)) {
      result(i) = tokens(position(i)).trim()
    }
    if (tokens.last == "positive") result(position.length - 1) = "1.0" else result(position.length - 1) = "0.0"

    result.mkString(",")
  }

  def parseLabeledPointString(str: String): LabeledPoint = {
    val value = str.split(",").map(element => element.toDouble)
    val features = value.slice(0, value.length - 1)
    val label = value.last
    LabeledPoint(label, Vectors.dense(features))
  }

  def parseStringLabeledPoint(instance: Array[Double], delimiter: String, minorityClass: String): String = {
    val features = instance.mkString(delimiter)
    (features + delimiter + minorityClass)
  }

  def parseStringMajoritaryClass(conv: Array[Map[String, Double]], line: String, delimiter: String): String = {

    val tokens = line.split(",")
    require(tokens.length == conv.length)

    val arr = (conv, tokens).zipped.map { (c, elem) =>
      c.getOrElse(elem.trim, elem.trim.toDouble)
    }

    val features = arr.slice(0, arr.length - 1)
    (features.mkString(delimiter) + delimiter + tokens.last)
  }
}