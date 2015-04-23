import csv
from string import digits

# NLTK
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# SKLearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from pprint import pprint

# Gensim
from gensim import corpora, models

trainingData = []
testingData = []

rows = 0

with open('reutersCSV.csv', 'rb') as fp:
	reader = csv.DictReader(fp)
	lemmatizer = WordNetLemmatizer()

	for line in reader:

		# ignore any not-used cases
		if line["purpose"] == "not-used":
			continue

		if line["purpose"] == "test":
			continue

		# if rows == 2:
		# 	break

		rows += 1

		if rows % 10 == 0:
			print(rows)

		# tokenize the text and remove any punctuation
		tokenizer = RegexpTokenizer(r'\w+')

		# concatenate title and text
		sentence = line["doc.title"] + ' ' + line["doc.text"]

		# remove any numbers
		sentence = sentence.translate(None, digits)

		tokens = tokenizer.tokenize(sentence)

		# run part of speech analysis
		tuples = nltk.pos_tag(tokens);

		# run lemmatization
		tuples = [ ( lemmatizer.lemmatize(p[0]), p[1] ) for p in tuples ]

		# remove the stopwords
		filteredWords = [x for x in tuples if not x[0] in stopwords.words('english')]

		# named entity recogniser
		filteredWords = nltk.ne_chunk(tagged_tokens=filteredWords)

		# concatenate the words into one sentence
		processedWords = [w.label() if type(w) is nltk.tree.Tree else w[0] for w in filteredWords]
		sentence = ' '.join(x for x in processedWords)

		# create instance of sentence for each topic
		for key in line:
			if(key.startswith('topic')):
				if(line[key] == '1'):
					if line["purpose"] == "train":
						tuple = (sentence, key[6:])
						trainingData.append(tuple)
					elif line["purpose"] == "test":
						tuple = (sentence, key[6:])
						testingData.append(tuple)

with open('trainingData.csv', 'w') as fp:
	writer = csv.writer(fp, delimiter=',')

	for i in trainingData:
		writer.writerow([i[0], i[1]])



with open('trainingData.csv', 'rb') as fp:
	reader = csv.reader(fp)

	for line in reader:
		if line[0].strip() != "":

			# only consider the 10 most popular topics
			classes = ["earn", "acquisitions", "money.fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]

			if line[1] in classes:
				trainingData.append(line)


with open('testingData.csv', 'rb') as fp:
	reader = csv.reader(fp)

	for line in reader:
		if line[0].strip() != "":

			# only consider the 10 most popular topics
			classes = ["earn", "acquisitions", "money.fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]

			if line[1] in classes:
				testingData.append(line)


### CREATE FEATURES ###

# create the topic model
trainingDataTokens = [x[0].split() for x in trainingData]
dictionary = corpora.Dictionary(trainingDataTokens)
corpus = [dictionary.doc2bow(token) for token in trainingDataTokens]
lda = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=40)

# extract the vocabulary from the topic model to prevent overfitting
vocabulary = []

for i in range(0, lda.num_topics - 1):
	topic = lda.show_topic(i, 10)

	for t in topic:
		vocabulary.append(t[1])

vocabulary = list(set(vocabulary))

corpus = [x[0] for x in trainingData]

# create a count feature representation
countVectorizer = CountVectorizer(min_df=1, vocabulary=vocabulary)
frequencyMatrix = countVectorizer.fit_transform(corpus)

# create a td*idx feature representation
tfidfVectorizer = TfidfVectorizer(min_df=1, vocabulary=vocabulary)
tfidfMatrix = tfidfVectorizer.fit_transform(corpus)



### CREATE CLASSIFIERS ###

# isolate training class
trainingDataClass = [x[1] for x in trainingData]

def naiveBayes(featureMatrix, predictionMatrix, output=True):
	naiveBayes = MultinomialNB()
	naiveBayesClassifier = naiveBayes.fit(featureMatrix, trainingDataClass)

	# predict naive bayes classifier
	# naiveBayesPrediction = naiveBayesClassifier.predict(predictionMatrix)
	naiveBayesPrediction = cross_validation.cross_val_predict(naiveBayesClassifier, predictionMatrix, trainingDataClass, cv=10)

	return naiveBayesPrediction


def randomForest(featureMatrix, predictionMatrix, output=True):
	randomForest = RandomForestClassifier(n_estimators=30)
	randomForestClassifier = randomForest.fit(featureMatrix, trainingDataClass)

	# predict
	# randomForestPrediction = randomForestClassifier.predict(featureMatrix)
	randomForestPrediction = cross_validation.cross_val_predict(randomForestClassifier, predictionMatrix, trainingDataClass, cv=10)

	return randomForestPrediction


def svm(featureMatrix, predictionMatrix, output=True):
	svm = LinearSVC()
	svmClassifier = svm.fit(featureMatrix, trainingDataClass)

	# svmPrediction = svmClassifier.predict(featureMatrix)
	svmPrediction = cross_validation.cross_val_predict(svmClassifier, predictionMatrix, trainingDataClass, cv=10)

	return svmPrediction


# analyse classifiers
def analyseClassifier(originalData, predictedData):

	# precision
	microPrecision = metrics.precision_score(originalData, predictedData, average="micro")
	macroPrecision = metrics.precision_score(originalData, predictedData, average="macro")

	# accuracy
	accuracy = metrics.accuracy_score(originalData, predictedData)

	# recall
	microRecall = metrics.recall_score(originalData, predictedData, average="micro")
	macroRecall = metrics.recall_score(originalData, predictedData, average="macro")

	fmeasure = metrics.f1_score(originalData, predictedData)

	print "Micro Precision: %f\nMacro Precision: %f" % (microPrecision, macroPrecision)
	print "Accuracy: %f" % (accuracy)
	print "Micro Recall: %f\nMacro Recall: %f" % (microRecall, macroRecall)
	print "F-measure: %f" % (fmeasure)


### ANALYSE CLASSIFIERS ###

# analyse naive bayes classifer
print "Naive Bayes Frequency Matrix"
analyseClassifier(trainingDataClass, naiveBayes(frequencyMatrix, frequencyMatrix))

print "\nNaive Bayes TF*IDF Matrix"
analyseClassifier(trainingDataClass, naiveBayes(tfidfMatrix, tfidfMatrix))

# analyse random forest classifier
print "Random Forest Frequency Matrix"
analyseClassifier(trainingDataClass, randomForest(frequencyMatrix, frequencyMatrix));

print "\nRandom Forest TF*IDF Matrix"
analyseClassifier(trainingDataClass, randomForest(tfidfMatrix, tfidfMatrix));

# analyse svm classifier
print "SVM Frequency Matrix"
analyseClassifier(trainingDataClass, svm(frequencyMatrix, frequencyMatrix))

print "\nSVM TF*IDF Matrix"
analyseClassifier(trainingDataClass, svm(tfidfMatrix, tfidfMatrix))



### CLUSTERING ###

# load all data
allData = []

with open('trainingData.csv', 'rb') as fp:
	reader = csv.reader(fp)

	for line in reader:
		if line[0].strip() != "":
			allData.append(line)


with open('testingData.csv', 'rb') as fp:
	reader = csv.reader(fp)

	for line in reader:
		if line[0].strip() != "":
			allData.append(line)

# create the topic model
adtrainingDataTokens = [x[0].split() for x in allData]
addictionary = corpora.Dictionary(adtrainingDataTokens)
adcorpus = [addictionary.doc2bow(token) for token in adtrainingDataTokens]
adlda = models.ldamodel.LdaModel(adcorpus, id2word=addictionary, num_topics=40)

# extract the vocabulary from the topic model to prevent overfitting
advocabulary = []

for i in range(0, adlda.num_topics - 1):
	topic = adlda.show_topic(i, 10)

	for t in topic:
		advocabulary.append(t[1])

advocabulary = list(set(advocabulary))

adcorpus = [x[0] for x in allData]

# create a count feature representation
adcountVectorizer = CountVectorizer(min_df=1, vocabulary=advocabulary)
adfrequencyMatrix = adcountVectorizer.fit_transform(adcorpus)

allDataClass = [x[1] for x in allData]

kmeans = KMeans(n_clusters=len(set(allDataClass)), n_init=1, init='k-means++')
kmeansCluster = kmeans.fit(adfrequencyMatrix)

print "KMeans"
print "Homogeneity: %f" % (metrics.homogeneity_score(allDataClass, kmeansCluster.labels_))
print "Completeness: %f" % (metrics.completeness_score(allDataClass, kmeansCluster.labels_))



dbscan = DBSCAN(eps=0.3, min_samples=10)
dbCluster = dbscan.fit(adfrequencyMatrix)

print "DBSCAN"
print "Homogeneity: %f" % (metrics.homogeneity_score(allDataClass, dbCluster.labels_))
print "Completeness: %f" % (metrics.completeness_score(allDataClass, dbCluster.labels_))


agg = AgglomerativeClustering(n_clusters=10, affinity='euclidean')
aggCluster = agg.fit(adfrequencyMatrix.toarray())

print "Agg"
print "Homogeneity: %f" % (metrics.homogeneity_score(allDataClass, aggCluster.labels_))
print "Completeness: %f" % (metrics.completeness_score(allDataClass, aggCluster.labels_))
