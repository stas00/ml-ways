# Natural Language Processing

Dense notes on NLP.

## Abbreviations

More extensive lists of abbreviations [1](https://github.com/AgaMiko/machine-learning-acronyms), [2](https://machinelearning.wtf/acronyms/)

```
ASGD    Averaged Stochastic Gradient Descent
AWD-LSTM  ASGD Weight-Dropped LSTM
BERT    Bidirectional Encoder Representations from Transformers
BPE     Byte Pair Encoding
BiLM    Bidirectional Language Model
CBOW    Continuous Bag-Of-Words
CFG     Context-free Grammar
CL      Computational Linguistics
CVT     Cross-View Training
CoLA    Corpus of Linguistic Acceptability
CoVe    Contextual Word Vectors
CRF     Conditional Random Field
DAG     Directed Acyclic Graph
DAE     Denoising Auto-Encoder
DCN     Dynamic Coattention Network
DCNN    Dynamic Convolutional Neural Network
DMN     Dynamic Memory Network
EDA     Exploratory Data Analysis
ELMo    Embeddings from Language Model
ESA     Explicit Semantic Analysis
FGN     Fine-Grained NER
FOL     First-Order Logic
GAN     Generative Adversarial Network
GEC     Grammatical Error Correction
GPT     Generative Pre-training Transformer
GRU     Gated-Recurrent Network
GloVe   Global Vectors for Word Representation
HAL     Hyperspace Analogue to Language
HDP     Hierarchical Dirichlet Process
IE      Information Extraction
IR      Information Retrieval
LDA     Latent Dirichlet Allocation
LSA     Latent Semantic Analysis (Truncated SVD)
LSI     Latent Semantic Indexing
LSTM    Long Short-Term Memory
MAE     Mean Absolute Error
MLM     Mask Language Model
MNLI    Multi-Genre NLI
MRPC    MicRosoft Paraphrase Corpus
MSE     Mean Squared Error
MaxEnt  Maximum Entropy (classifier) (softmax)
NER     Named-Entity Recognition
NLG     Natural Language Generation
NLI     Natural Language Inference (Text Entailment)
NLP     Natural Language Processing
NLU     Natural Language Understanding
NMT     Neural Machine Translation
NTN     Neural Tensor Network
NiN     Network-in-network (1x1 convconnections)
PCFG    Probabilistic Context Free Grammar
POS     Parts-Of-Speech
QRNN    Quasi-Recurrent Neural Networks
QNLI    Question NLI
RACE    ReAding Comprehension from Examinations
RMSE    Root Mean Squared Error
RNN     Recurrent Neural Network
RNN     Recursive Neural Network
RNTN    Recursive Neural Tensor Network
RP      Random Projections
RTE     Recognizing Textual Entailment (now called NLI)
SG      Skip-Gram
SNLI    Stanford Natural Language Inference
SOTA    State-Of-The-Art
SQuAD   Stanford Question Answering Dataset
SRL     Semantic Role Labeling
SST     Stanford Sentiment Treebank
STLR    Slanted Triangular Learning Rates
SWAG    Situations With Adversarial Generations
TDNN    Time-Delayed Neural Network
TF      Term­Frequency
TF­IDF  Term­Frequency­Inverse­Document­Frequency
TLM     Translation Language Modeling
ULMFiT  Universal Language Model Fine-tuning
USE     Universal Sentence Encoder
VAE     Variational Autoenconder
VSM     Vector Space Model
WSD     Word Sense Disambiguation
ZSL     Zero-Shot Learning
t-SNE   t-distributed Stochastic Neighbor Embedding



```

## Glossary and Terminology

**Denotational semantics**: The concept of representing an idea as a symbol (a word or a one-hot vector). It is sparse and cannot capture similarity. This is a "localist" representation.

**Distributional semantics**: The concept of representing the meaning of a word based on the context in which it usually appears. It is dense and can better capture similarity.

**Distributional similarity**: similar words have similar context.

**Transformer** is an architecture for transforming one sequence into another one with the help of two parts (Encoder and Decoder).

**Constituency Parsing** is a way to break a piece of text (e.g. one sentence) into sub-phrases. One of the goals of constituency parsing (also known as "phrase structure parsing") is to identify the constituents in the text which would be useful when extracting information from text. By knowing the constituents after parsing the sentence, it is possible to generate similar sentences that are syntactically correct.

**Lemmas** are root forms of words.

**Named Entity Recognition**: which words in a sentence are a proper name, organization name, or entity?

**Textual Entailment**: given two sentences, does the first sentence entail or contradict the second sentence?

**Coreference Resolution**: given a pronoun like “it” in a sentence that discusses multiple objects, which object does “it” refer to?



### Word Embedding Models

Popular off-the-shelf word embedding models:

- Word2Vec (by Google)
- GloVe (by Stanford)
- fastText (by Facebook)


#### Word2vec

- 2 algorithms: continuous bag-of-words (CBOW) and skip-gram. CBOW aims to predict a center word from the surrounding context in terms of word vectors. Skip-gram does the opposite, and predicts the distribution (probability) of context words from a center word.

- 2 training methods: negative sampling and hierarchical softmax. Negative sampling defines an objective by sampling negative examples, while hierarchical softmax defines an objective using an efficient tree structure to compute probabilities for all the vocabulary.



## Augmentation


https://amitness.com/2020/05/data-augmentation-for-nlp/


## Metrics



### Perplexity

Perplexity is often used as an intrinsic evaluation metric for gauging how well a language model can capture the real word distribution conditioned on the context.

A [perplexity](https://en.wikipedia.org/wiki/Perplexity) of a discrete probability distribution pp is defined as the exponentiation of the entropy:

$2^{H(p)} = 2^{-\sum_x p(x) \log_2 p(x)}$

Given a sentence with $N$ words, $s = (w_1, \dots, w_N)$, the entropy looks as follows, simply assuming that each word has the same frequency, $\frac{1}{N}$:

$H(s) = -\sum_{i=1}^N P(w_i) \log_2  p(w_i)  = -\sum_{i=1}^N \frac{1}{N} \log_2  p(w_i)$

The perplexity for the sentence becomes:

$
2^{H(s)} = 2^{-\frac{1}{N} \sum_{i=1}^N \log_2  p(w_i)}
= (2^{\sum_{i=1}^N \log_2  p(w_i)})^{-\frac{1}{N}}
= (p(w_1) \dots p(w_N))^{-\frac{1}{N}}
$

A good language model should predict high word probabilities. Therefore, the smaller perplexity the better.


### Compilations

* cs224u: [Evaluation metrics in NLP](https://github.com/cgpotts/cs224u/blob/master/evaluation_metrics.ipynb)
* scikit: [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)


## Functions

### Softmax

After applying softmax, each component will be in the interval (0, 1) and the total will add up to 1, so that they can be interpreted as probabilities.

The larger input components will correspond to larger probabilities.

**Temperature** is used to scale the logits before applying softmax. (logits/τ)

1. For high temperatures (τ → ∞), all components have nearly the same probability and the lower the temperature, the more expected values affect the probability. This results in more diversity and also more mistakes.

2. When the temperature is 1, the softmax is computed on unscaled logits.

3. For a low temperature (τ → 0), the probability of the action with the highest expected value tends to 1. Larger logit values makes softmax more confident, but also more conservative in its samples (it is less likely to sample from unlikely candidates).

https://cs.stackexchange.com/a/79242/113823






## Linguistics

- **[Hyponymy](https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy)** && **Hypernymy**: a hyponym is a word or phrase whose semantic field is included within that of another word. San Francisco (hyponym) is an **instance of** a city (hypernym). A pigeon is a hyponym of bird; which, in turn, is a hyponym of animal. A bird is a hypernym of a pigeon. An animal is a hypernym of a bird.

- **Antonymy**: acidic is the **opposite** of basic

- **Meronymy**: an alternator is a **part of** a car

- **[Polysemy](https://en.wikipedia.org/wiki/Polysemy)** is the capacity for a word or phrase to have multiple meanings, usually related by contiguity of meaning within a semantic field. e.g. crane: (n) machine, (n) bird, (v) to strain out one's neck.

[Semantic change](https://en.wikipedia.org/wiki/Semantic_change) (also semantic shift, semantic progression, semantic development, or semantic drift) is a form of language change regarding the evolution of word usage—usually to the point that the modern meaning is radically different from the original usage.



### Monotonicity reasoning

**Monotonicity**. A system is monotonic if it grows without shrinking.

**Monotonicity reasoning** is a type of reasoning based on word replacement, requires the ability to capture the interaction between lexical and syntactic structures. Consider examples in (1) and (2).

   (1)  a. All     [    workers ↓] joined for a [French dinner ↑]
        b. All     [    workers  ] joined for a [       dinner  ]
        c. All     [new workers  ] joined for a [French dinner  ]

   (2)  a. Not all [new workers ↑] joined for a dinner
        b. Not all [    workers  ] joined for a dinner

A context is **upward entailing** (shown by [... ↑]) that allows an inference from (1a) to (1b), where *French dinner* is replaced by a more general concept *dinner*. On the other hand, a **downward entailing** context (shown by [... ↓]) allows an inference from (1a) to (1c), where *workers* is replaced by a more specific concept *new workers*. Interestingly, the direction of monotonicity can be reversed again by embedding yet another downward entailing context (e.g., *not* in (2)), as witness the fact that (2a) entails (2b). To properly handle both directions of monotonicity, NLI models must detect monotonicity operators (e.g., all, not) and their arguments from the syntactic structure.
(this excerpt is from [Can neural networks understand monotonicity reasoning?](https://arxiv.org/abs/1906.06448))




## Libraries

Useful libraries and modules:

- [Annoy](https://github.com/spotify/annoy) (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings to search for points in space that are close to a given query point.

## Transformers

- Huggingface [transformers](https://github.com/huggingface/transformers). The main transformers library.

- [Simple Transformers](https://github.com/ThilinaRajapakse/simpletransformers). Transformers made simple with training, evaluation, and prediction possible with one line each

- [AdaptNLP](https://github.com/Novetta/adaptnlp). A high level framework and library for running, training, and deploying state-of-the-art Natural Language Processing (NLP) models for end to end tasks. Built on top of Zalando Research's Flair and Hugging Face's Transformers.

- [spacy-transformers](https://github.com/explosion/spacy-transformers) provides spaCy model pipelines that wrap Hugging Face's transformers package, so you can use them in spaCy.

## Good Paper Explanations

- AWD-LSTM: Average-SGD Weight-Dropped LSTM
   * [What makes the AWD-LSTM great](https://yashuseth.blog/2018/09/12/awd-lstm-explanation-understanding-language-model/)

- ULMFiT: Universal Language Model Fine Tuning
   * [Understanding the Working of ULMFiT](https://yashuseth.blog/2018/06/17/understanding-universal-language-model-fine-tuning-ulmfit/)


## fastai NLP notebooks

- seq2seq:
   * https://github.com/ohmeow/seq2seq-pytorch-fastai/blob/master/seq2seq-rnn-attn.ipynb


## Sources

- [Stanford cs224n](http://web.stanford.edu/class/cs224n/)
- [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#metric-perplexity)

## Books

- Chris Manning and Hinrich Schuetze - Foundations of Statistical Natural Language Processing


## Newsletters

- [NLP Newsletter](https://github.com/dair-ai/nlp_newsletter)
- [NLP News](http://newsletter.ruder.io/)
