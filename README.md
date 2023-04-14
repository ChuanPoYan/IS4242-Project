# Project Description
## Investigating Supervised Machine Learning Methods to Understand Factors that Affects Number of Shares of Blog Posts
With the growing number of media content creators, it has become increasingly important to understand the factors that contribute to the success of posts in order to stand out. Predicting the number of times that articles are shared is useful as it enables them to understand their audience better and tailor content to meet their preferences. This is especially important as many content creators generate revenue through advertising or affiliate marketing. Thus, accurately predicting the number of views for blog articles can provide valuable insights and help content creators achieve their goals.

## Table of Contents
* [Dependencies](#Dependencies)
* [Data Extraction](#Data-Extraction)
* [Feature Engineering](#Feature-Engineering)
* [Model Design](#Model-Design)


# Dependencies

Python libraries that the notebooks depend on can be installed using:

```
$ pip install -r requirements.txt
```

# Data Extraction
Data is first extracted from the main page of TheSmartLocal.com, where articles’ preview is being shown . The articles are sorted in descending order of its published date and has 10 different articles per page. Using BeautifulSoup, the code loops through 439 pages and extract information for each article which includes the actual article link. This data is then stored in a dataframe named article_preview. A second data extraction is then performed using BeautifulSoup by looping through each article link stored in article_preview and extracting information within each article link. The final dataset consists of 4390 rows and 13 columns. 

## Description of the raw Data Extracted

- **url**: url of article
- **title**: title of article
- **subcategory**: first subcategory of article
- **preview**: preview content of article (before clicking into article)
- **content**: full content of article (includes image credits)
- **reading_duration**: number of minutes to read entire article, around 200 words per minute
- **author**: author of article
- **publish_date**: publication date of article
- **num_imgs**: number of images in article
- **num_hrefs**: number of hyperlinks in article
- **num_self_hrefs**: number of hyperlinks in article linked to thesmartlocal.com
- **num_tags**: number of tags at the end of the article
- **num_shares**: number of shares of article

# Feature Engineering
There are 4 main sections:
1. Feature Engineering
   1. Main Categories
   2. Time Features
   3. Article Count
   4. Text Features
   5. Feature Crossing
3. Data Preprocessing
   1. One-Hot-Encoding for Categorical Variables
   2. Binning for Target Variable
5. Feature Selection
6. Text Embedding 
   1. Embedding Techniques
   2. Models for Evaluation
   3. Dimensionality Reduction

# Model Design
There are 3 main sections:
1. Defining Utility Functions
   1. Generating the Cross Validation scores
   2. Evaluating the model using confusion matrix
   3. Finding the best parameters through GridSearchCV
2. Loading of Dataset
   1. 2 Datasets
      1. Tree Based Ensemble Models - With Categorical Variables
      2. Other Models - Categorical Variables has been processed with One Hot Encoding and Feature Selection
   2. Scaling of b to run probabilistic and instance-based algorithms 
3. Exploration of Models
   1. Probabilistic
      1. Logistic Regression
      2. Naive Bayes
   2. Instance Based
      1. K-Nearest Neighbours
      2. Support Vector Machine
   3. Tree Based
      1. Random Forest
      2. Catboost
      3. LightGBM
      4. Adaboost
   4. Deep Learning
      1. Convolutional Neural Network

For each model, a baseline model is created to obtain the cross validation score. The performance of the model is evaluated using the confusion matrix and accuracy. To optimise the parameters, GridSearchCV is find the parameters that produces a better performance. The best model is then evaluated once again using the confusion matrix and accuracy.

The best performing model is the **Support Vector Machine** with an **accuracy of 48.7%** after parameter optimisation.


# Authors
* Chuan Po Yan
* Clement Goh Junhui
* Lim Si Tian
* Milton Sia
* Ong Yi En
* Samuel Lim Zhao Xuan
* Tan Si Jing
