package utils

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.{Vector => OldVector}

import scala.collection.mutable.ArrayBuffer
//import org.apache.spark.mllib.linalg.Vector
//import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.feature.LabeledPoint


/** K Nearest Neighbors algorithms. 
  *
  * @param train        Traning set
  * @param k            Number of neighbors
  * @param distanceType Distance.Manhattan or Distance.Euclidean
  *                     //@param numClass Number of classes
  * @author sergiogvz
  */
class KNN(val train: ArrayBuffer[(Long, LabeledPoint)], val k: Int, val distanceType: Distance.Value) {

  /** Calculates the k nearest neighbors.
    * Para un `x` calcula la distancia contra todos los del training set y guarda los mas cercanos (sin que sea él mismo).
    * Así completa el array `nearest` y `distA`
    *
    * @param x Test sample
    * @return Distance and id of each nearest neighbors
    */
  def neighbors(x: (Long, LabeledPoint)): (Long, Array[(Long, Float)]) = {
    var nearest = Array.fill(k)(-1)
    var distA = Array.fill(k)(0.0f)
    val size = train.length

    for (i <- 0 until size) { //for instance of the training set
      val dist: Float = euclidean(x._2.features, train(i)._2.features)
      if (dist > 0d) { //leave-one-out control
        var stop = false
        var j = 0
        while (j < k && !stop) { //Checks if it can be inserted as NN
          if (nearest(j) == (-1) || dist <= distA(j)) { //está vacío o la distancia actual es menor, inserto!
            //muevo los elem una posición a la derecha
            for (l <- ((j + 1) until k).reverse) { //for (int l = k - 1; l >= j + 1; l--)
              nearest(l) = nearest(l - 1)
              distA(l) = distA(l - 1)
            }
            nearest(j) = i
            distA(j) = dist
            stop = true
          }
          j += 1
        }
      }
    }
    //val key = x._1.toString() + "," + x._1.toString()
    val key = x._1
    var out: Array[(Long, Float)] = new Array[(Long, Float)](k) // idNeighbour, distance
    for (i <- 0 until k) out(i) = (train(nearest(i))._1, distA(i))


    (key, out) //return (currentPoint, Array[(idneigh1, distidneigh1), (idneigh2, distidneigh2), ... ,(idneighk, distidneighk)]
  }

  def euclidean(x: Vector, y: Vector) = {
    var sum = 0.0
    val size = x.size

    for (i <- 0 until size) sum += (x(i) - y(i)) * (x(i) - y(i))

    Math.sqrt(sum).toFloat

  }

}

/** Factory to compute the distance between two instances.
  *
  * @author sergiogvz
  */
object Distance extends Enumeration {
  val Euclidean, Manhattan = Value

  /** Computes the (Manhattan or Euclidean) distance between instance x and instance y.
    * The type of the distance used is determined by the value of distanceType.
    *
    * @param x            instance x
    * @param y            instance y
    * @param distanceType type of the distance used (Distance.Euclidean or Distance.Manhattan)
    * @return Distance
    */
  //def apply(x: Vector, y: Vector, distanceType: Distance.Value) = {
  def apply(x: OldVector, y: OldVector, distanceType: Distance.Value) = {
    distanceType match {
      case Euclidean => euclidean(x, y)
      case Manhattan => manhattan(x, y)
      case _ => euclidean(x, y)
    }
  }

  /** Computes the Euclidean distance between instance x and instance y.
    * The type of the distance used is determined by the value of distanceType.
    *
    * @param x instance x
    * @param y instance y
    * @return Euclidean distance
    */
  //private def euclidean(x: Vector, y: Vector) = {
  private def euclidean(x: OldVector, y: OldVector) = {
    var sum = 0.0
    val size = x.size

    for (i <- 0 until size) sum += (x(i) - y(i)) * (x(i) - y(i))

    Math.sqrt(sum).toFloat

  }

  /** Computes the Manhattan distance between instance x and instance y.
    * The type of the distance used is determined by the value of distanceType.
    *
    * @param x instance x
    * @param y instance y
    * @return Manhattan distance
    */
  //private def manhattan(x: Vector, y: Vector) = {
  private def manhattan(x: OldVector, y: OldVector) = {
    var sum = 0.0
    val size = x.size

    for (i <- 0 until size) sum += Math.abs(x(i) - y(i))

    sum.toFloat
  }

}