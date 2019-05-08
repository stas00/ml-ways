# Natural Language Processing



## Abbreviations

```
ASGD    Averaged Stochastic Gradient Descent
AWD-LSTM  ASGD Weight-Dropped LSTM
BERT    Bidirectional Encoder Representations from Transformers
BPE     Byte Pair Encoding
BiLM    Bidirectional Language Model
CBOW    Continuous Bag-Of-Words
CFG     Context-free Grammar
CVT     Cross-View Training
CoLA    Corpus of Linguistic Acceptability
CoVe    Contextual Word Vectors
DCN     Dynamic Coattention Network
DCNN    Dynamic Convolutional Neural Network
DMN     Dynamic Memory Network
ELMo    Embeddings from Language Model
ESA     Explicit Semantic Analysis
FGN     Fine-Grained NER
GAN     Generative Adversarial Network
GPT     Generative Pre-training Transformer
GRU     Gated-Recurrent Network
GloVe   Global Vectors for Word Representation
HDP     Hierarchical Dirichlet Process
LDA     Latent Dirichlet Allocation
LSA     Latent Semantic Analysis
LSTM    Long Short-Term Memory
MLM     Mask Language Model
MNLI    Multi-Genre NLI
MRPC    MicRosoft Paraphrase Corpus
NER     Named-Entity Recognition
NLG     Natural Language Generation
NLI     Natural Language Inference (Text Entailment)
NLP     Natural Language Processing
NLU     Natural Language Understanding
NMT     Neural Machine Translation
PCFG    Probabilistic Context Free Grammar
POS     Parts-Of-Speech
QNLI    Question NLI
RACE    ReAding Comprehension from Examinations
RNN     Recurrent Neural Network
RNN     Recursive Neural Network
RNTN    Recursive Neural Tensor Network
RP      Random Projections
RTE     Recognizing Textual Entailment
SG      Skip-Gram
SNLI    Stanford Natural Language Inference
SOTA    State-Of-The-Art
SQuAD   Stanford Question Answering Dataset
SST     Stanford Sentiment Treebank
STLR    Slanted Triangular Learning Rates
SWAG    Situations With Adversarial Generations
Srl     Semantic Role Labeling
TDNN    Time-Delayed Neural Network
ULMFiT  Universal Language Model Fine-tuning
VAE     Variational Autoenconder
WSD     Word Sense Disambiguation
```

## Glossary and Terminology

Denotational semantics: The concept of representing an idea as a symbol (a word or a one-hot vector). It is sparse and cannot capture similarity. This is a "localist" representation.

Distributional semantics: The concept of representing the meaning of a word based on the context in which it usually appears. It is dense and can better capture similarity.

Distributional similarity: similar words have similar context.

Transformer is an architecture for transforming one sequence into another one with the help of two parts (Encoder and Decoder).

Constituency Parsing is a way to break a piece of text (e.g. one sentence) into sub-phrases. One of the goals of constituency parsing (also known as "phrase structure parsing") is to identify the constituents in the text which would be useful when extracting information from text. By knowing the constituents after parsing the sentence, it is possible to generate similar sentences that are syntactically correct.



### Word Embedding Models

Popular off-the-shelf word embedding models:

- Word2Vec (by Google)
- GloVe (by Stanford)
- fastText (by Facebook)


#### Word2vec

- 2 algorithms: continuous bag-of-words (CBOW) and skip-gram. CBOW aims to predict a center word from the surrounding context in terms of word vectors. Skip-gram does the opposite, and predicts the distribution (probability) of context words from a center word.

- 2 training methods: negative sampling and hierarchical softmax. Negative sampling defines an objective by sampling negative examples, while hierarchical softmax defines an objective using an efficient tree structure to compute probabilities for all the vocabulary.




## Metrics

### Perplexity

Perplexity is often used as an intrinsic evaluation metric for gauging how well a language model can capture the real word distribution conditioned on the context.

A [perplexity](https://en.wikipedia.org/wiki/Perplexity) of a discrete probability distribution $p$ is defined as the exponentiation of the entropy:

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


# Sources

- http://web.stanford.edu/class/cs224n/
- https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#metric-perplexity
