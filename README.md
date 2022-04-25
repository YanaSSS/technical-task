clustering-documents-tfidf-kmeans.ipynb

# Articles Clusterization

## Abstract
The file _docs.json_ has 300 documents that have to be grouped properly based on their content. The task for clustering documents presupposes the use of unsupervised machine learning since we don't have labeled data. 
I will start with pre-processing of the data. This includes data exploration and cleaning, extracting significant data using TF-IDF vectorizer and finally Kmeans algorithm, paired with _inertia_ criterion to assess performance of the algorithm.

## Methodology
### Data exploration
Below is a snapshot of the content (columns) of the dataset:
![image](https://user-images.githubusercontent.com/43813983/165172795-5ef0439e-3db8-442a-a7ec-29bee088ac74.png)

Before proceeding with anything else I need to edit the columns in order to navigate easily through data. Several columns will be removed too since they are of no use to the task (having too many null values, date columns, or columns containing only one value:
- columns where over 90 percent of the values are null - 23 to 32
- columns _index, type, score, and version_ since they contain a single value which cannot be used to group the data.
- _media_outlet_logo_ column will not be used since it contains jpg filenames that do not give useful information for grouping the articles.
- the date columns: _date_created_, _air_date_, _time_stamp_ are also irrelevant to the task.
- the columns _full_text_ and _entity_name_ could be useful but more than one third and respectively more than half of the data is null in these columns. Since it is not a numerical data I cannot impute mean or duplicate values. I cannot afford to drop the rows either since I will be left with very little data to work with.
- there are only two rows with missing data in _prospect_keyword_ column, so these rows can be dropped.
- also do not need _state_ and _country_ (content of the article is what matters)
- as for _article_type_ missing values I will explore the possibility to add value based on simillarity with other columns.
- _media_type_ and _commentator_ columns are linked to the respective _media_type_id_ and _commentator_id_. The id columns will be dropped.

_media_type_, _program_name_, _article_type_, and _source_system_ columns were set as category type.

![image](https://user-images.githubusercontent.com/43813983/165176233-a94dd825-e340-4b79-a926-a7259d062c8a.png)

![image](https://user-images.githubusercontent.com/43813983/165176301-78b80266-caaa-4046-bae4-13da928c3145.png)

![image](https://user-images.githubusercontent.com/43813983/165176350-55379512-9221-4704-93c5-8055ac54c887.png)

![image](https://user-images.githubusercontent.com/43813983/165176181-35f7b71b-ec80-454e-8e1e-764bcb6db271.png)


**article_type** contained several null values. After grouping data by _article_type_, _media_type_, and _source_system_ and doing visual assessment of the group, the best approach seemed to be replacing the NaNs with an article type name: 'Social Media':

![image](https://user-images.githubusercontent.com/43813983/165175461-50323d9a-3937-42c0-8455-ac2e71237c13.png)

This is the final result of the pre-processing described above (since I am not editing the column content, this stage is added to exploration and not to cleaning):
![image](https://user-images.githubusercontent.com/43813983/165175982-4ddee575-ccc3-4136-a0c8-8d916fec7a59.png)

For the current version of the task I will not need all columns that were left after the first stage of pre-processing.

### Data cleaning
In this stage I am going to prepare the _abstract_ column content for vectorization and clustering.
A function **clean_text** is added to pre-process text, generate tokens and return cleaned text. Several python libraries are used here:
```
import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
```
The process of cleaning contains the following steps:
1. Transform the input into a string and lowercase it
2. Remove links from text
3. Remove digits and words containing digits
4. Remove multiple spaces, tabs, and line breaks
5. Remove ellipsis characters
6. Replace dashes between words with a space 
7. Remove punctuation
8. Remove new line char
9. Tokenize text 
10. Remove tokens using a list of stop words
11. Remove too short tokens
12. Join tokens in to return cleaned text

![image](https://user-images.githubusercontent.com/43813983/165178237-7cac8009-7a13-4cda-b37a-c9c73900ef45.png)

During this process I have noticed there are duplicate abstracts in the data and proceeded with droping them. As a result I was left with a dataframe with shape (275,20).

### Tf-Idf Vectorization
TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection  or corpus. The tf-idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general (https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

In this project sklearn [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) is use where the following parameters are edited:
- sublinear_tf
- max_df
- min_df

### Dimensionality Reduction
[Principal Component Analysis (PCA)](https://builtin.com/data-science/step-step-explanation-principal-component-analysis) is used for dimensionality reductioon. It is a  method used to reduce the dimensionality of large data sets by transforming them into smaller ones and still keeping most of the information from the large data set. 

Reducing the number of variables comes at the expense of accuracy. It is basically a trade-off between accuracy and simplicity. Having less dimensions (in our case 2) helps visualizing and analyzing data much easier and faster for ML algorithms.

PCA is also used here because of the _inertia_ criterion. The latter is used to measure of how internally coherent KMeans clusters are.

One drawback of this criterion is it is a not normalized metric. As explained in scikit learn [documentation](https://scikit-learn.org/stable/modules/clustering.html) "lower values are better and zero is optimal. But in very high-dimensional spaces, Euclidean distances tend to become inflated (this is an instance of the so-called “curse of dimensionality”). Running a dimensionality reduction algorithm such as Principal component analysis (PCA) prior to k-means clustering can alleviate this problem and speed up the computations." 

### Implementation of K-means

The K-means algorithm works only with numbers this is why the process of vectorizing the text is crucial.

This algorithm uses partitioning method for clustering i.e., dividing the data set with n objects into k partitions. Its main objective is to minimize the distance between the points within a cluster i.e., to minimize the sum of distances between the points in the cluster and their respective centroid.

It is suitable for finding similarities between entities based on distance measures with small datasets. Kmeans assumes spherical shapes of clusters. It does not work well when clusters are in different shapes. 

When the clusters are evaluated their main qualities should be taken into account. First, all data points in a cluster should be similar to each other. Second, the data points between different clusters should be as different as possible. In regards to the first I may use _inertia_. It calculates the sum of distances of all the points withing a cluster from the centroid of that cluster (intracluster distance).

On the other hand the second property can be addressed by calculating the [Dunn Index](https://pyshark.com/dunn-index-for-k-means-clustering-evaluation-using-python/) which takes into account the distance between two clusters.

I want to minimize the _inertia_ but I also want to maximize _Dunn index_.

After several tests the number of clusters was set to 6 where _inertia\__ is ~1.05.

### Giving proper names of clusters
For the purpose of defining proper cluster names `vectorizer.get_feature_names_out()` is used. I chose top 10 keywords for each cluster.

```
cluster_map = {
    0: 'volvo', 
    1: 'business', 
    2: 'asian politics',
    3: 'australian air transport', 
    4: 'airplane deals', 
    5: 'financial',
}
```
### Visualization

The names that I gave are not optimal as are not the clusters. The reason for that is obvious when I plot the scatter. As it was already mentioned K-means does not like non-spherical clusters. 

Clusters 0, 3, and 5 look better structured while 1, 2, and 4 visually do not have the desired shape.

### Improvements
1. The dataset has some significant drawbacks (besides being small which possibly leads to overfitting). On one hand there are articles which are in language other than english (seemed like turkish). For this reason adding stopwords in this language may be useful.
- Adding other stopwords, including country names may also have good effect on clusterization.
Another issue is that because of a lot of missing data I could not use the full article text but only the abstract.
- In this case, instead of droping the _full_text_ column I can leave it and join it with the abstract column. This way I will have at least half of the articles with full text which can improve the clusterization process.

2. In relation to vectorizer:
- defining n_gram ranges should be tested
- I may further compare results by substituting tf-idf with word2vec

3. Calculating Dunn index can give further clarificatio of the results. 

4. Exploring alternatives to K-means in order to deal with unusual cluster shapee (Possibly [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) which is based on similarity related to the density of data points.

5. More attention to _prospect_keywords_. Possibly comparing them with results from the clustering.
