# zlearn
This is a wrapper around SciKit Learn's Random Forest Classifier (rfc) routines. Given an OTU
table and known classes for some of the samples, train a rfc on all the data and get the
output: how well could the samples be classified, what's the out-of-bag error, and what are
the important features?
#Version
v-0.1.0
### Input
You'll need an **OTU table** in QIIME format (i.e., OTUs on the rows, samples on the columns). You'll also need a **classes file** that specifies what samples belong to what class.

The classes file can have two formats: two-column or one-column. In one-column format, each line specifies a sample-class pair, like

```
sampleA1	healthy
sampleA2	healthy
sampleB1	disease
sampleB2	disease
```

In two-column format, the first line of a block is the class name and the rest of the lines are sample names, like

```
# you can put in a comment
healthy
sampleA1
sampleA2

disease
sampleB1
sampleB2
```
### Usages
For adabooster,you may directly run pepline as follow:
```
zlearn.py otutab txt design txt normalize tag ab_e 4 ab n 1000000
```
For randomforest,
```
zlearn.py otutab txt design txt normalize tag rf_e 4 rf n 1000000
```
##At least ten billion times was recommended for each classfier.
### Options
The command-line arguments will show you what kinds of options you can pass to the random forest classifier. Of special note in `-n`, the number of trees in the forest.

Make sure you put the `--help` *after* you've specified the classifier (viz., `rf` or `ab`) so that you can see the options specific to that classifier.

### Output
All the output files are given a tag to help organize the output. By default, a tag is produced that is specific to that OTU table and class file. (It's a hex digest of the concatenation of the two files.)

File name | Contents
----------|------------
tag_classes.txt | the true classes that you specified
tag_cmd.txt | the command you input to run slime2
tag_featimp.txt | feature importances. column1 is the OTUs; col2 is the important; col3 is the cumulative importance
tag_params.txt | the parameters used when making the RFC
tag_results.txt | an explicit confusion matrix. if a sample was correctly classified, the line is "--". if a sample of true class X was misclassified as class Y, you get ">> X misclassified as Y".
tag_rfc.pkl | a pickled python object. it's the RFC object, which has a bunch of extra data attached to it.
tag_scores.txt | the "mean" score (i.e., the fraction of samples correctly classified by the RFC when trained on all data) and the out-of-bag score.

### Common gotcha's
- If you get a `ValueError`, check the format of your classes file. Output from `R` can be problematic, since by default it writes the row numbers, which means you end up with too many columns.
