#CountVectorizer for text classification
#It's time to begin building your text classifier! The data has been loaded into a DataFrame called df. 
#Explore it in the IPython Shell to investigate what columns you can use. The .head() method is particularly informative.

#In this exercise, you'll use pandas alongside scikit-learn to create a sparse text vectorizer you can use to train and 
#test a simple supervised model. To begin, you'll set up a CountVectorizer and investigate some of its features.

# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split 

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size = 0.33, random_state = 53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words="english")

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])

##########################################################################################

#TfidfVectorizer for text classification
#Similar to the sparse CountVectorizer created in the previous exercise, you'll work on creating 
#tf-idf vectors for your documents. You'll set up a TfidfVectorizer and investigate some of its features.

#In this exercise, you'll use pandas and sklearn along with the same X_train, y_train and X_test, y_test 
#DataFrames and Series you created in the last exercise.

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])

###########################################################################

#Inspecting the vectors
#To get a better idea of how the vectors work, you'll investigate them by converting them into pandas DataFrames.

#Here, you'll use the same data structures you created in the previous two exercises (count_train, count_vectorizer, 
#tfidf_train, tfidf_vectorizer) as well as pandas, which is imported as pd.

# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=count_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(tfidf_df.columns) - set(count_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))

##############################################################

#Training and testing the "fake news" model with CountVectorizer
#Now it's your turn to train the "fake news" model using the features you identified and extracted. 
#In this first exercise you'll train and test a Naive Bayes model using the CountVectorizer data.

#The training and test sets have been created, and count_vectorizer, count_train, and count_test have been computed.

#Import the metrics module from sklearn and MultinomialNB from sklearn.naive_bayes.
#Instantiate a MultinomialNB classifier called nb_classifier.
#Fit the classifier to the training data.
#Compute the predicted tags for the test data.
#Calculate and print the accuracy score of the classifier.
#Compute the confusion matrix. To make it easier to read, specify the keyword argument labels=['FAKE', 'REAL']

# Import the necessary modules
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train,y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)

#####################################################################

#Training and testing the "fake news" model with TfidfVectorizer
#Now that you have evaluated the model using the CountVectorizer, you'll 
#do the same using the TfidfVectorizer with a Naive Bayes model.

#The training and test sets have been created, and tfidf_vectorizer, tfidf_train, 
#and tfidf_test have been computed. Additionally, MultinomialNB and metrics have been imported from, 
#respectively, sklearn.naive_bayes and sklearn.

#Instantiate a MultinomialNB classifier called nb_classifier.
#Fit the classifier to the training data.
#Compute the predicted tags for the test data.
#Calculate and print the accuracy score of the classifier.
#Compute the confusion matrix. As in the previous exercise, specify the keyword argument labels=['FAKE', 'REAL'] 
#so that the resulting confusion matrix is easier to read.

# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)

####################################################################

#Improving your model
#Your job in this exercise is to test a few different alpha levels using the Tfidf vectors to determine if 
#there is a better performing combination.

#The training and test sets have been created, and tfidf_vectorizer, tfidf_train, and tfidf_test have been computed.
#Create a list of alphas to try using np.arange(). Values should range from 0 to 1 with steps of 0.1.
#Create a function train_and_predict() that takes in one argument: alpha. The function should:
#Instantiate a MultinomialNB classifier with alpha=alpha.
#Fit it to the training data.
#Compute predictions on the test data.
#Compute and return the accuracy score.
#Using a for loop, print the alpha, score and a newline in between. Use your train_and_predict() function to compute the score. 
#Does the score change along with the alpha? What is the best alpha?

# Create the list of alphas: alphas
alphas = np.arange(0, 1, 0.1)

# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train,y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test,pred)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()
    
#############################################################

#Inspecting your model
#Now that you have built a "fake news" classifier, you'll investigate what it has learned. 
#You can map the important vector weights back to actual words using some simple inspection techniques.

#You have your well performing tfidf Naive Bayes classifier available as nb_classifier, and the vectors as tfidf_vectorizer.

#Save the class labels as class_labels by accessing the .classes_ attribute of nb_classifier.
#Extract the features using the .get_feature_names() method of tfidf_vectorizer.
#Create a zipped array of the classifier coefficients with the feature names and sort them by the coefficients. To do this, first use zip() with the arguments nb_classifier.coef_[0] and feature_names. Then, use sorted() on this.
#Print the top 20 weighted features for the first label of class_labels.
#Print the bottom 20 weighted features for the second label of class_labels.

# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])





