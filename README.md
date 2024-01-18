# CS 412 CONGABONGAZONGA BROJECT

## 1. Overall Structure

The project's architecture is illustrated in the following schema:

![Schema](diagram.drawio.png)

This diagram demonstrates the different steps of the project we followed in order to come up with this system. 

## 2. Main Contributions to the Project

### 2.1 Direct Clustering

In contrast to the existing approach, which relied on cosine similarity for clustering, we adopted direct clustering techniques. This involved experimenting with various clustering algorithms, such as K-Means, DBScan, and GMM, to determine the most effective method for our specific problem.

### 2.2 Preprocessing

Recognizing the limitations of the existing project's word data preprocessing, we implemented a preprocessing step to address issues like differentiating between similar words ('change' and 'changed'). This ensured a more accurate representation of the data for subsequent analysis.

### 2.3 Keyword Lists

We introduced two types of keyword lists, namely blacklist and whitelist, along with their respective frequencies. This refinement helped enhance the quality of features used in the model.

### 2.4 Sentiment Scores

Sentiment scores were incorporated as a new feature to provide insights into user satisfaction with the answers. This addition aimed to capture the emotional context of the text data.

### 2.5 Vectorization Techniques

Instead of relying solely on TF-IDF, we explored alternative vectorization approaches, including Bag of Words and Word2Vec. This experimentation allowed us to identify the most suitable vectorization method for our specific use case.

### 2.6 Dimensionality Reduction with PCA

Recognizing the high dimensionality of NLP data, we applied Principal Component Analysis (PCA) to reduce dimensionality. Different PCA values were experimented with, and the optimal value, providing over 90% explained variance, was determined to enhance clustering effectiveness.

## 3. Preprocessing

In the existing project, preprocessing was not applied to the data. However, recognizing its importance in frequency-based approaches, we implemented preprocessing to address issues related to word variations and improve the overall model performance.

## 4. Vectorizer

While the existing project solely utilized the TF-IDF approach, we expanded our vectorization techniques to include Bag of Words and Word2Vec. This allowed us to assess the effectiveness of different vectorization methods in capturing semantic relationships within the data.

## 5. Clustering

Clustering played a pivotal role in our approach. We experimented with various clustering techniques, such as K-Means, DBScan, and GMM, and assessed their effectiveness using the Silhouette score. Dimensionality reduction was later explored to improve clustering outcomes. By doing so, we were able to classify each prompt to the specific question.

## 6. PCA

Given the high dimensionality inherent in NLP data, we applied PCA to reduce dimensionality. Through experimentation, we identified the optimal number of PCA components that provided over 90% explained variance, leading to improved clustering performance, as indicated by the increased Silhouette score. The reason we utilized PCA is to provide a more concise and meaningful representation of complex datasets, our dataset is complex, offering benefits such as improved model simplicity and efficiency.

## 7. Feature Engineering

We introduced sentiment scores as additional features to gauge user satisfaction with responses. For this, we experimented with two sentiment models which are Bert and TextBlob. After the experimentation, we observed that we were more satisfied with the TextBlob.
Furthermore, the application of whitelist and blacklist words during feature engineering provided finer control over the model's understanding of relevant terms. The reasoning behind the Blacklist and Whitelist approach is that checking every single word's relation with the prompt is not suitable. Instead, we thought that keeping the words that have negative meanings on the prompt in the Blacklist and keeping the words that have positive meanings on the prompt in the Whitelist is better. For instance, the word "gini" is included in the Whitelist since it has a positive effect on the score directly. If the student does not mention "gini" explicitly in his/her prompt, then s/he lost points for sure.
