name := "smote-bd"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies += "com.typesafe" % "config" % "1.2.1"

resolvers ++= Seq("Akka Repository" at "http://repo.akka.io/releases/")

resolvers ++= Seq("Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases")

spName := "majobasgall/smote-bd" // the name of the Spark Package
sparkVersion := "2.2.0" // the Spark Version the package depends on.
sparkComponents += "mllib" // creates a dependency on spark-mllib.