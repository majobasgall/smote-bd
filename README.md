# SMOTE-BD
It is a fully scalable preprocessing approach for imbalanced classification in Big Data. It is based on one of the most widespread preprocessing solutions for imbalanced classification, namely the SMOTE [1] algorithm, which creates new synthetic instances according to the neighborhood of each example of the minority class.

## How to run it?

A generic example to run it could be:

```spark-submit --master "URL" --executor-memory "XG" "path-to-jar".jar --class "path-to-main" --datasetName="aName" --headerFile="path-to-header" --inputFile="path-to-input" --delimiter=", " --outputPah="path-to-output" --seed="aSeed" --K="number-of-neighbours" --numPartitions="number-of-parts"  --nReducers="number-of-reducers" --numIterations="number-of-iterations" --minClassName="min-class-name" -overPercentage=100 ```

- Parameters of spark: ```--master "URL" | --executor-memory "XG" ```. They can be usefull for launch with diferent settings and datasets.
- ```--class path.to.the.main aJarFile.jar``` Determine the jar file to be run.
- ```datasetName``` The name of the current dataset.
- ```headerFile``` Full path to header file.
- ```inputFile``` Full path to input file.
- ```delimiter``` Delimiter of each attribute value.
- ```outputPah``` Full path to output directory.
- ```seed``` A seed to generate random numbers.
- ```K``` Number of nearest neighbours.
- ```numPartitions``` Number of partitions to split data.
- ```nReducers``` Number of reducers (required by the K-NN stage).
- ```numIterations``` Number of iterations (required by the K-NN stage).
- ```minClassName``` Name of the minority class (according to the header file).
- ```overPercentage``` Percentage of balancing between classes.

## Please, cite this software as:
Basgall, M. J., Hasperué, W., Naiouf, M., Fernández, A., & Herrera, F. (2018). SMOTE-BD: An Exact and Scalable Oversampling Method for Imbalanced Classification in Big Data. Journal of Computer Science and Technology, 18(03), e23. https://doi.org/10.24215/16666038.18.e23

### Bibtex:
@article{Basgall_Hasperué_Naiouf_Fernández_Herrera_2018,  
	title={SMOTE-BD: An Exact and Scalable Oversampling Method for Imbalanced Classification in Big Data},  
	volume={18},   
	url={http://journal.info.unlp.edu.ar/JCST/article/view/1122},   
	DOI={10.24215/16666038.18.e23},   
	number={03},   
	journal={Journal of Computer Science and Technology},   
	author={Basgall, María José and Hasperué, Waldo and Naiouf, Marcelo and Fernández, Alberto and Herrera, Francisco},   
	year={2018},   
	month={Dec.},   
	pages={e23}   
}

## References
[1] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. J. Artif. Int. Res., 16(1), 321–357.
