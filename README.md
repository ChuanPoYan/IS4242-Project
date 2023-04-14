# Project Description

## Investigating Supervised Machine Learning Methods to Understand Factors that Affects Number of Shares of Blog Posts

With the growing number of media content creators, it has become increasingly important to understand the factors that contribute to the success of posts in order to stand out. **Predicting the number of times that articles are shared** is useful as it enables them to understand their audience better and tailor content to meet their preferences. This is especially important as many content creators generate revenue through advertising or affiliate marketing. Thus, accurately predicting the number of views for blog articles can provide valuable insights and help content creators achieve their goals.

## Table of Contents

- [Dependencies](#Dependencies)
- [Data Extraction](#Data-Extraction)
- [Feature Engineering](#Feature-Engineering)
- [Model Design](#Model-Design)

# Dependencies

Certain segments of this project uses specific package versions. Any deviation of those versions listed in the _requirements.txt_ file may lead to errors while running the scripts. To install the correct packages and their respective versions, use:

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

## For Reproducibility

To reproduce the results of data extraction, run the notebook `The Smart Local Dataset Creation.ipynb`. The data extracted by this is included in the repository under `dataset/SmartLocal/smartlocal_raw.parquet`.

Warning: Extraction process has a long running time due to `robots.txt` requirements of TSL.

# Feature Engineering

Data Analysis
1. Feature Engineering
   1. Main Categories
   2. Time Features
   3. Article Count
   4. Text Features
   5. Feature Crossing
2. Data Preprocessing
   1. One-Hot-Encoding for Categorical Variables
   2. Binning for Target Variable
3. Feature Selection
4. Text Embedding
   1. Embedding Technique
   2. Dimensionality Reduction

Text Embedding Selection
1. Text Embedding
   1. Trained Word2Vec
   2. Trained Doc2Vec
   3. TF-IDF
   4. BERT
2. Evaluation

The text embedding selection is split out as a separate section to reduce length of the file. It is still part of the feature engineering process.

## For Reproducibility

To reproduce the results of feature engineering, run the notebook `The Smart Local Data Analysis.ipynb`. Data created from the notebook `The Smart Local Dataset Creation.ipynb` is needed.

The data created by this are included in the repository under `dataset/SmartLocal/`.
| Data Name | Description |
|-----------|-------------|
| `smartlocal_text.csv` | Contains cleaned text data, used in text embedding selection |
| `title_embeddings_cbow.csv` | Contains the values of title embeddings using Word2Vec CBOW |
| `preview_embeddings_cbow.csv` | Contains the values of preview embeddings using Word2Vec CBOW |
| `content_embeddings_cbow.csv` | Contains the values of content embeddings using Word2Vec CBOW |
| `X_train.csv` | Contains the feature values of the training set with categorical data one-hot encoded, used in non tree based models |
| `X_train_tree.csv` | Contains the feature values of the training set without categorical data one-hot encoded, used in tree based models |
| `X_test.csv` | Contains the feature values of the testing set with categorical data one-hot encoded, used in non tree based models |
| `X_test_tree.csv` | Contains the feature values of the testing set without categorical data one-hot encoded, used in tree based models |
| `y_train.csv` | Contains the target variable of the training set, used in non tree based models |
| `y_train_tree.csv` | Contains the target variable of the training set, used in tree based models |
| `y_test.csv` | Contains the target variable of the testing set, used in non tree based models |
| `y_test_tree.csv` | Contains the target variable of the testing set, used in tree based models |

To reproduce the results of text embedding selection, run the notebook `The Smart Local Text Embedding Selection.ipynb`. Data created from the notebook `The Smart Local Data Analysis.ipynb` is needed. 

Additionally, two files used here '/dataset/Pretrained Embedding Model/GoogleNews-vectors-negative300.bin' and './dataset/Pretrained Embedding Model/glove.6B.100d.txt' are not found in the GitHub as their size are too big. Instead they can be downloaded here:
- https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300
- https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt

Warning: Embedding process has a long running time.

The embeddings created by this are included in the repository under `dataset/text_embedding/`. They are further separated by the folders `title`, `preview`, and `content`.

| Embedding Name | Description |
|----------------|-------------|
| `emb_sg_train.csv`  | Contains training set values of embeddings using Word2Vec Skip-Gram |
| `emb_sg_test.csv`   | Contains testing set values of embeddings using Word2Vec Skip-Gram |
| `emb_cbow_train.csv`  | Contains training set values of embeddings using Word2Vec CBOW |
| `emb_cbow_test.csv`   | Contains testing set values of embeddings using Word2Vec CBOW |
| `emb_ggl_train.csv`  | Contains training set values of embeddings using Word2Vec Google |
| `emb_ggl_test.csv`   | Contains testing set values of embeddings using Word2Vec Google |
| `emb_glove_train.csv`  | Contains training set values of embeddings using Word2Vec GloVe |
| `emb_glove_test.csv`   | Contains testing set values of embeddings using Word2Vec GloVe |
| `emb_dbow_train.csv`  | Contains training set values of embeddings using Doc2Vec DBOW |
| `emb_dbow_test.csv`   | Contains testing set values of embeddings using Doc2Vec DBOW |
| `emb_dm_train.csv`  | Contains training set values of embeddings using Doc2Vec DM |
| `emb_dm_test.csv`   | Contains testing set values of embeddings using Doc2Vec DM |
| `emb_dbow_dm_train.csv`  | Contains training set values of embeddings using Doc2Vec DBOW & DM |
| `emb_dbow_dm_test.csv`   | Contains testing set values of embeddings using Doc2Vec DBOW & DM |
| `emb_tfidf_train.npz`  | Contains training set values of embeddings using TF-IDF |
| `emb_tfidf_test.npz`   | Contains testing set values of embeddings using TF-IDF |
| `emb_bert_train.npz`  | Contains training set values of embeddings using BERT |
| `emb_bert_test.npz`   | Contains testing set values of embeddings using BERT |

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

## For Reproducibility

To reproduce the results of data extraction, run the notebook `The Smart Local Modelling.ipynb`. Data created from the notebook `The Smart Local Data Analysis.ipynb` are needed.

Warning: Model tunning process has a long running time.

The models generated by this are included in the repository under `dataset/Model/`.
| Model Name | Description |
|------------|-------------|
| `lr_best_model.pkl` | Fitted Logistic Regression Model with parameter tunning |
| `nb_best_model.pkl` | Fitted Naive Bayes Model with paramter tunning |
| `knn_best_model.pkl` | Fitted K-Nearest Neighbours Model with parameter tunning |
| `svm_best_model.pkl` | Fitted Support Vector Machine Model with parameter tunning |
| `rf_best_model.pkl` | Fitted Random Forest Model with parameter tunning |
| `cb_best_model.pkl` | Fitted CatBoost Model with parameter tunning |
| `lgb_best_model.pkl` | Fitted LightGBM with parameter tunning |
| `ada_best_model.pkl` | Fitted AdaBoost Model with parameter tunning |
| `cnn_model.pkl` | Fitted Convolutional Neural Networks Model |

# Authors

- Chuan Po Yan
- Clement Goh Junhui
- Lim Si Tian
- Milton Sia
- Ong Yi En
- Samuel Lim Zhao Xuan
- Tan Si Jing
