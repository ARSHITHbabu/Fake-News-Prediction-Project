# Fake-News-Prediction-Project
A project based on  Fake News Prediction which utilizes machine learning to analyze textual features and classify news articles as either genuine or fake.Fake News Prediction projects are essential for addressing the growing challenge of misinformation. These projects contribute to building tools that can assist in identifying and mitigating the impact of fake news, thereby fostering a more informed public discourse.

**GETTING STARTED**

1)*IMPORTING THE LIBRARIES*:
At first we have to imports essential Python libraries for text analysis and machine learning.It includes NumPy and Pandas for data manipulation, NLTK for natural language processing, Seaborn and Matplotlib for visualization, and scikit-learn for machine learning tasks such as text vectorization and linear regression modeling.

      1)NumPy (np): It is a library for numerical operations in Python which provides support for large, multi-dimensional arrays and matrices, along with mathematical functions.

      2)Pandas (pd): It is a data manipulation library which provides data structures like DataFrames, that are useful for handling and analyzing structured data.

      3)Re (re): This is a regular expression module in Python which is used for pattern matching and manipulation of strings.

      4)NLTK (Natural Language Toolkit): A library for working with human language data. It includes various tools for tasks such as tokenization, stemming, tagging, parsing, and more.

      5)Seaborn (sns): A data visualization library based on Matplotlib, providing an interface for creating informative and attractive statistical graphics.

      6)Matplotlib.pyplot (plt): Matplotlib is a 2D plotting library for Python and Pyplot is a module within Matplotlib that provides a convenient interface for creating various types of plots.

      7)Stopwords: A module from NLTK that contains a list of common words (e.g., "we," "must," "our") that are often removed from text data during preprocessing because they don't contribute much to the meaning.

      8)PorterStemmer: It is a stemming algorithm from NLTK which is used to reduce words to their root/base form and it helps in normalizing words and reducing dimensionality in text data.

      9)TfidfVectorizer: A class from scikit-learn used for converting a collection of raw documents to a matrix of TF-IDF features. TF-IDF stands for Term Frequency-Inverse Document Frequency and is commonly used in text analysis.

      10)Train-test split (train_test_split): A function from scikit-learn that splits a dataset into training and testing sets, which is crucial for evaluating machine learning models.

      11)LogisticRegression: It is a class from scikit-learn which is used for logistic regression.It is also a common algorithm for binary classification tasks.

      12)Accuracy_score: A function from scikit-learn for computing the accuracy of a classification model.


2)*IMPORTING THE DATASETS*:
Here,we are going to import 2 datasets from kaggle which is needed for the project.The 2 datasets are Fake.csv and True.csv.Here we can execute commands specifying the respective Kaggle dataset paths for each and use Pandas to read them into DataFrames in your Colab environment.


3)*EDA (exploratory data analysis*:
EDA is a crucial step in the process of data analysis where data analyst or data scientist visually and statistically explore datasets to understand their main characteristics. They use techniques like summary statistics, visualizations, and data cleaning etc.EDA helps to uncover patterns, relationships, and potential outliers, guiding subsequent modeling and decisions for the analysis. EDA simplifies a deeper comprehension of the data's structure and aids in formulating hypotheses for more advanced analyses.
In this project,we have done the EDA for the given datasets by finding their information,number of null values,plotting graph using seaborn and finding the percentage of true news and fake news in the datasets.


4)*DATA PRE-PROCESSING*:
It is a fundamental step in data analysis and machine learning that involves cleaning and transforming raw data into a format suitable for analysis or modeling. Tasks in this phase include handling missing values, removing outliers, standardizing or normalizing features, encoding categorical variables, and scaling data. By addressing irregularities and enhancing data quality, pre-processing ensures that subsequent analyses or machine learning algorithms can effectively extract meaningful insights from the dataset.

      i)Preprocessing:It is a crucial phase in data analysis and machine learning which involves cleaning and transforming raw data into a suitable format for further analysis or modeling.This tasks include the methods of handling missing values, scaling features, encoding categorical variables, and removing outliers, ensuring that the data is well-prepared and optimized for subsequent analysis or machine learning algorithms.

      ii)Stemming:Stemming is a text normalization technique in natural language processing that involves reducing words to their root or base form, known as the "stem." This process helps to eliminate variations of a word, reducing different inflections or derivations to a common form.

      iii)TfidfVectorizer(Term Frequency-Inverse Document Frequency Vectorizer): It is a feature extraction method in natural language processing and information retrieval. This technique converts a collection of raw documents into a numerical format by representing each document as a vector. It takes into account both the frequency of terms in a document and their rarity across all documents, assigning higher weights to terms that are important in a specific document but not overly common across the entire corpus.


4)*MODEL FITTING*:
It involves training a machine learning model on a dataset, adjusting its parameters to capture patterns and relationships.The fitting process involves optimization algorithms that iteratively adjust the model's parameters to minimize the difference between the predicted outcomes and the actual outcomes in the training data.

      i)Linear Regression:It is a statistical method for modeling the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data.

      ii)DecisionTreeClassifier:It is a machine learning algorithm used for classification tasks.This algorithm recursively splits the dataset based on features, aiming to create decision rules that optimize classification accuracy.

      iii)Logistic Regression:It is a classification algorithm used for binary and multi-class classification tasks.It helps to model the probability of an instance belonging to a particular class by applying a logistic or sigmoid function to a linear combination of input features

      iv)Confusion Matrix: It is a table used in classification to evaluate the performance of a machine learning model. It summarizes the count of true positive, true negative, false positive, and false negative predictions, providing a comprehensive view of a model's accuracy, precision, recall, and F1 score

