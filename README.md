Disaster Tweets Classification using Deep Learning
==================================================

1\. Introduction
----------------

This project focuses on developing a Natural Language Processing (NLP) model to classify tweets as either relating to real disasters (target=1) or not (target=0). In emergency response systems, the ability to automatically identify disaster-related communications on social media can be crucial for timely interventions and resource allocation.

Text classification is a fundamental NLP task that involves assigning predefined categories to text documents based on their content. The challenge in this project lies in correctly interpreting short, informal text (tweets) that may contain slang, abbreviations, hashtags, and other social media-specific language patterns.

2\. Dataset Overview
--------------------

The dataset comes from the Kaggle "Natural Language Processing with Disaster Tweets" competition and consists of:

-   **Training set**: 7,613 tweets with labels (disaster or non-disaster)
-   **Test set**: 3,263 tweets without labels (to be predicted)

Each entry contains:

| Feature | Description |
| --- | --- |
| id | Unique identifier for each tweet |
| keyword | Keywords extracted from the tweet (may be null) |
| location | Location the tweet was sent from (may be null) |
| text | The actual content of the tweet |
| target | Binary classification (1 = disaster, 0 = non-disaster) |

The task is to build a model that can accurately classify tweets in the test set as either disaster-related or not.

3\. Exploratory Data Analysis (EDA)
-----------------------------------

Exploratory Data Analysis helps us understand the dataset's characteristics, identify patterns, and inform our preprocessing and modeling strategies.

### 3.1 Data Loading and Initial Inspection

We begin by loading the training and test datasets and examining their basic properties. The training set contains 7,613 tweets with 5 columns, while the test set contains 3,263 tweets with 4 columns (excluding the target variable).

### 3.2 Examining Text Length Distribution

Understanding the distribution of text lengths is important for determining appropriate padding strategies for our neural networks.

Looking at the descriptive statistics:

-   Mean text length: ~101 characters
-   Median text length: ~107 characters
-   Maximum text length: 157 characters

The histogram shows that most tweets are between 70-140 characters long, which is consistent with Twitter's character limits at the time these tweets were collected. The similarity in text length distributions between training and test sets suggests consistency in the data.

### 3.3 Target Variable Analysis

The target variable distribution shows:

-   4,342 non-disaster tweets (57.0%)
-   3,271 disaster tweets (43.0%)

While there is a slight class imbalance, it's not severe enough to require techniques like oversampling. However, we'll use class weights in our model training to account for this imbalance.

### 3.4 Keyword and Location Analysis

The dataset includes keyword and location fields that could potentially provide additional context:

-   Top keywords include disaster-related terms like "fatalities," "deluge," and "armageddon"
-   Top locations include "USA," "New York," and "United Kingdom"

However, a significant number of entries have missing values for these fields.

### 3.5 Missing Values Analysis

The missing values analysis shows:

-   Training set: 61 missing keywords (0.8%), 2,533 missing locations (33.3%)
-   Test set: 26 missing keywords (0.8%), 1,105 missing locations (33.9%)

Given the high percentage of missing location data, we'll focus on the text content for our classification model rather than incorporating location information.

4\. Data Preprocessing
----------------------

Based on our EDA, we'll now preprocess the text data to prepare it for deep learning models.

### 4.1 Text Cleaning

Text data often contains noise and irrelevant information. Our cleaning process includes:

1.  **Removing URLs**: Web links don't usually contribute to disaster classification.
2.  **Removing mentions**: Twitter handles (@username) are not informative for our task.
3.  **Removing non-alphabetic characters**: Special characters, numbers, and punctuation are filtered out.
4.  **Converting to lowercase**: Standardizes the text and reduces vocabulary size.
5.  **Removing stop words**: Common words like "the," "is," and "and" that don't contribute much meaning.

The `clean_text` function above implements these steps and returns a list of tokens (words) for each tweet.

### 4.2 Word Frequency Analysis

Analyzing the most frequent words in each class can provide insights into distinguishing features:

**Top words in disaster tweets**:

-   Disaster-specific terms: "fire," "disaster," "suicide," "killed," "crash"
-   Location references: "California," "Hiroshima"
-   Emergency services: "police"

**Top words in non-disaster tweets**:

-   General expressions: "like," "just," "will," "get"
-   Personal pronouns: "im," "dont," "cant"
-   Positive sentiment: "love"

This analysis confirms that disaster tweets contain specific vocabulary related to emergencies, catastrophes, and tragic events, while non-disaster tweets use more casual, everyday language.

### 4.3 Tokenization and Padding

To feed text data into neural networks, we need to convert words to numerical form:

1.  **Tokenization**: Converting each word to a unique integer index using Keras' `Tokenizer`.
2.  **Sequence creation**: Transforming each tweet into a sequence of integers.
3.  **Padding**: Ensuring all sequences have the same length by adding zeros to shorter sequences.

The maximum sequence length is determined by the longest tweet in our training set, ensuring all information is preserved.

After preprocessing, we split our training data into training and validation sets (80/20 split) to monitor model performance during training.

5\. Model Architecture Design
-----------------------------

For text classification tasks, particularly with sequential data like tweets, recurrent neural networks (RNNs) are well-suited because they can capture contextual information and relationships between words. We'll implement and compare two different RNN architectures:

### 5.1 Bidirectional GRU Architecture

Gated Recurrent Units (GRUs) are a simplified variant of LSTMs that maintain performance while being computationally more efficient. Key components of our GRU model:

1.  **Embedding Layer**: Converts word indices into dense vectors of fixed size, capturing semantic relationships between words.
2.  **Bidirectional GRU Layer**: Processes sequences in both forward and backward directions, capturing context from both past and future words.
3.  **Dense Layer with Regularization**: Applies L2 regularization to prevent overfitting.
4.  **Dropout Layer**: Randomly deactivates neurons during training to reduce overfitting.
5.  **Output Layer**: Single neuron with sigmoid activation for binary classification.

GRUs are particularly effective for shorter sequences like tweets because they:

-   Require fewer parameters than LSTMs
-   Can capture short-term dependencies efficiently
-   Are faster to train

### 5.2 Stacked Bidirectional LSTM Architecture

Long Short-Term Memory (LSTM) networks are designed to address the vanishing gradient problem in traditional RNNs, making them capable of learning long-term dependencies. Our stacked LSTM model includes:

1.  **Embedding Layer**: Similar to the GRU model.
2.  **Multiple Bidirectional LSTM Layers**: Stacked to enable hierarchical feature extraction.
3.  **Dense Layer with Regularization**: Prevents overfitting.
4.  **Dropout Layer**: Adds further regularization.
5.  **Output Layer**: Binary classification output.

Stacking multiple LSTM layers allows the model to:

-   Learn more complex patterns and hierarchical representations
-   Capture both short and long-term dependencies
-   Potentially achieve higher accuracy for complex relationships

Both architectures use bidirectional processing, which is crucial for text classification as it allows the model to understand context from both preceding and following words in a sentence.

6\. Model Implementation and Hyperparameter Tuning
--------------------------------------------------

Finding the optimal hyperparameters is crucial for maximizing model performance. We use Keras Tuner for systematic hyperparameter optimization.

### 6.1 GRU Architecture Hyperparameter Tuning

For the GRU architecture, we tune the following hyperparameters:

| Hyperparameter | Search Range | Description |
| --- | --- | --- |
| Embedding Dimension | 50-300 | Size of the word embedding vectors |
| GRU Units | 32-128 | Number of units in the GRU layer |
| GRU Dropout | 0.2-0.5 | Dropout rate for the GRU layer |
| Recurrent Dropout | 0.2-0.5 | Dropout rate for recurrent connections |
| Dense Units | 32-128 | Number of units in the dense layer |
| L2 Regularization | 0.0001-0.01 | Strength of L2 regularization |
| Dense Dropout | 0.3-0.6 | Dropout rate for the dense layer |
| Learning Rate | 0.01-0.0001 | Step size for gradient descent optimization |

The `build_model_gru` function creates models with different hyperparameter combinations, and the RandomSearch algorithm explores the hyperparameter space to find the optimal configuration.

### 6.2 Stacked LSTM Architecture Hyperparameter Tuning

For the Stacked LSTM architecture, we tune similar hyperparameters:

| Hyperparameter | Search Range | Description |
| --- | --- | --- |
| Embedding Dimension | 50-300 | Size of the word embedding vectors |
| Number of LSTM Layers | 1-2 | Number of stacked LSTM layers |
| LSTM Units | 32-128 | Number of units in each LSTM layer |
| LSTM Dropout | 0.2-0.5 | Dropout rate for LSTM layers |
| Recurrent Dropout | 0.2-0.5 | Dropout rate for recurrent connections |
| Dense Units | 32-128 | Number of units in the dense layer |
| L2 Regularization | 0.0001-0.01 | Strength of L2 regularization |
| Dense Dropout | 0.3-0.6 | Dropout rate for the dense layer |
| Learning Rate | 0.01-0.0001 | Step size for gradient descent optimization |

Both tuning processes use:

-   30 trials with different hyperparameter combinations
-   Early stopping to prevent overfitting
-   Learning rate reduction when performance plateaus
-   Model checkpointing to save the best models

This systematic approach to hyperparameter optimization helps us find the best configuration for each architecture without manual trial and error.

7\. Model Training and Evaluation
---------------------------------

After identifying the optimal hyperparameters for each architecture, we proceed to train the models and evaluate their performance.

### 7.1 GRU Model Results

The best GRU model achieved the following hyperparameter configuration:

| Hyperparameter | Optimal Value |
| --- | --- |
| Embedding Dimension | 100 |
| GRU Units | 32 |
| GRU Dropout | 0.4 |
| Recurrent Dropout | 0.3 |
| Dense Units | 96 |
| L2 Regularization | 0.0031 |
| Dense Dropout | 0.3 |
| Learning Rate | 0.01 |

The GRU model training results show:

-   Best validation accuracy: 0.8089 (80.89%)
-   Early stopping occurred after 4 epochs due to increasing validation loss
-   Final training accuracy was very high (~97.84%), indicating some overfitting despite regularization

The loss and accuracy plots show typical patterns of overfitting, with training metrics continuing to improve while validation metrics plateau and then deteriorate.

### 7.2 Stacked LSTM Model Results

The best Stacked LSTM model had the following configuration:

| Hyperparameter | Optimal Value |
| --- | --- |
| Embedding Dimension | 300 |
| Number of LSTM Layers | 2 |
| LSTM Units (Layer 1) | 128 |
| LSTM Units (Layer 2) | 96 |
| LSTM Dropout | 0.3 |
| Recurrent Dropout | 0.2 |
| Dense Units | 32 |
| L2 Regularization | 0.0041 |
| Dense Dropout | 0.4 |
| Learning Rate | 0.001 |

The Stacked LSTM model results show:

-   Best validation accuracy: 0.8030 (80.30%)
-   Similar to the GRU model, early stopping activated after 4 epochs
-   Training accuracy reached ~96.49%, again showing signs of overfitting

Both models showed similar patterns of quick convergence on the training set but limited generalization to the validation set, suggesting that the complexity of disaster tweet classification presents challenges for perfect generalization.

8\. Architecture Comparison
---------------------------

Comparing the two architectures reveals interesting insights:

| Metric | GRU Model | Stacked LSTM Model |
| --- | --- | --- |
| Best Validation Accuracy | 80.89% | 80.30% |
| Model Size (parameters) | 1,490,765 | 5,092,709 |
| Convergence Speed | Faster | Slower |

Key observations:

1.  **Performance**: The GRU model slightly outperformed the Stacked LSTM model in terms of validation accuracy.
2.  **Efficiency**: The GRU model was significantly faster to train (approximately 20× faster).
3.  **Model Complexity**: The LSTM model had over 3× more parameters than the GRU model.
4.  **Training Behavior**: Both models showed similar patterns of rapid training set learning but limited validation set generalization.

These results align with research suggesting that GRUs can match or exceed LSTM performance for certain tasks while being computationally more efficient, especially for shorter sequences like tweets.

The models scored:

Stacked LTSM: .79
GRU: .78

9\. Discussion and Conclusion
-----------------------------

### 9.1 What Worked Well

1.  **Preprocessing Strategy**: The text cleaning approach effectively removed noise (URLs, mentions, special characters) and stop words, focusing on meaningful content.
2.  **Word Embeddings**: Using an embedding layer learned from our specific dataset rather than pre-trained embeddings worked well, allowing the model to capture domain-specific semantic relationships.
3.  **Bidirectional Approaches**: Both models benefited from bidirectional processing, capturing context from both directions in tweets.
4.  **GRU Efficiency**: The GRU architecture achieved slightly better performance with significantly fewer parameters and training time compared to the LSTM model.
5.  **Regularization Techniques**: Implementing dropout, recurrent dropout, and L2 regularization helped control overfitting to some extent.
6.  **Class Weights**: Accounting for the slight class imbalance through class weights ensured the model didn't bias toward the majority class.

### 9.2 Challenges Encountered

1.  **Overfitting**: Both models showed signs of overfitting despite regularization, with large gaps between training and validation performance.
2.  **Limited Contextual Understanding**: Tweet classification remains challenging because short texts provide limited context, and disaster-related terms can appear in non-disaster contexts.
3.  **Informal Language**: The informal nature of tweets, including abbreviations, slang, and unusual grammatical structures, presents challenges for NLP models.
4.  **Missing Contextual Information**: Tweets often reference external events or contain images that provide crucial context, which our text-only model couldn't access.

### 9.3 Future Improvements

1.  **Pre-trained Embeddings**: Using pre-trained word embeddings like GloVe or Word2Vec could provide better semantic representations, especially for rare words.
2.  **Transformer-based Models**: Implementing BERT, RoBERTa, or other transformer architectures could capture more complex relationships in the text.
3.  **Data Augmentation**: Techniques like back-translation or synonym replacement could increase training data diversity.
4.  **Ensemble Methods**: Combining predictions from multiple models could improve overall performance and robustness.
5.  **Feature Engineering**: Incorporating additional features like sentiment scores, named entity recognition, or keyword presence might enhance model performance.
6.  **Attention Mechanisms**: Adding attention layers could help the model focus on the most relevant parts of tweets for classification.
7.  **Cross-Validation**: Implementing k-fold cross-validation would provide more robust performance estimates.

### Conclusion

This project demonstrated the effectiveness of recurrent neural networks, particularly GRUs, for disaster tweet classification. The GRU model achieved a validation accuracy of 80.89%, outperforming the more complex Stacked LSTM model while requiring significantly less computational resources.

The results suggest that simpler, more efficient architectures may be preferable for short-text classification tasks like tweet analysis, especially in resource-constrained or real-time application scenarios. The project also highlights the challenges of social media text classification, where informal language, limited context, and implicit references can make accurate classification difficult even for sophisticated deep learning approaches.
