# Statistics


## Feature selection

**Mutual information** (MI) of term t and class c measures how much information the presence/absence of a term contributes to making the correct classification decision on c.

**X^2** is a measure of how much expected counts E and observed counts N deviate from each other. A high value of X^2 indicates that the hypothesis of independence, which implies that expected and observed counts are similar, is incorrect.

**Frequency-based feature selection** - selecting the terms that are most common in the class.


## Evaluation Metrics

The **precision** is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

The **recall** is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

The **F-beta** score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0. The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.

The **support** is the number of occurrences of each class in ground truth (correct) target values.

[scikit-learn.org](https://scikit-learn.org/stable/modules/model_evaluation.html)

A **macro-average** treats all classes equally - it computes the metric independently for each class and then takes the average.

A **micro-average** aggregates the contributions of all classes to compute the average metric. If there is a class imbalance this might be preferable over macro-average. (usually for multi-class classification)




## T-test, p-value

The **p-value** is the probability that the results from your sample data occurred by chance. Typical threshold p = 0.05 => 5% probability. Small p-value indicates that your hypotheses (feature) has a statistical significance.



## Probabilities

The **prior probabilities** are also called **class priors**, which describe ”the general probability of encountering a particular class.”


## Distributions

| Distribution | Categories | Number of trials | Example          |
| ------------ | ---------- | ---------------- | ---------------- |
| Bernoulli    | 2          | 1                | 1 coin toss      |
| Binomial     | 2          | many             | 2 heads 3 tosses |
| Multinoulli  | many       | 1                | 1 dice roll      |
| Multinomial  | many       | many             | 2 6s in 3 rolls  |


### Naive Bayes estimators

**Multi-variate Bernoulli Naive Bayes**. The binomial model is useful if your feature vectors are binary (i.e., 0s and 1s). One application would be text classification with a bag of words model where the 0s 1s are "word occurs in the document" and "word does not occur in the document"

**Multinomial Naive Bayes**. The multinomial naive Bayes model is typically used for discrete counts. E.g., if we have a text classification problem, we can take the idea of bernoulli trials one step further and instead of "word occurs in the document" we have "count how often word occurs in the document", you can think of it as "number of times outcome number x_i is observed over the n trials"

**Gaussian Naive Bayes**. we assume that the features follow a normal distribution. Instead of discrete counts, we have continuous features (e.g., the popular Iris dataset where the features are sepal width, petal width, sepal length, petal length).



## Classification

1. **Binary classification** or **binomial classification** is classifying instances into one of 2 classes
2. **Multiclass classification** is classifying instances into one of 3+ classes
3. **Multi-label classification** assigns one or more classes to a single instance.


## Books

- [Daniel Jurafsky, James H. Martin - Speech and Language Processing - An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://amzn.to/2PG3mJz)
- [Ian Goodfellow, Yoshua Bengio, Aaron Courville - Deep Learning - Adaptive Computation And Machine Learning](https://amzn.to/2sPHYIK)
- [Christopher D. Manning, Prabhakar Raghavan, Hinrich Schutze - Introduction to information retrieval](https://amzn.to/2Mdv5iv)
- [Yoav Goldberg - Neural Network Methods for Natural Language Processing (2017)](https://amzn.to/2sSKWw5)
