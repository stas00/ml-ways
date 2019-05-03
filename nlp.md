# Natural Language Processing



## Terminology

```
AWD-LSTM  ASGD Weight-Dropped LSTM, ASGD Averaged Stochastic Gradient Descent
BERT    Bidirectional Encoder Representations from Transformers
CBOW    Continuous Bag-Of-Words
CoLA    Corpus of Linguistic Acceptability
CoVe    Contextual Word Vectors
DCNN    Dynamic Convolutional Neural Network
ELMo    Embeddings from Language Model
FGN     Fine-Grained NER
GAN     Generative Adversarial Network
GRU     Gated-Recurrent Network
LSTM    Long Short-Term Memory
MNLI    Multi-Genre NLI
MRPC    MicRosoft Paraphrase Corpus
NER     Named-Entity Recognition
NLG     Natural Language Generation
NLI     Natural Language Inference (Text Entailment)
NLP     Natural Language Processing
NLU     Natural Language Understanding
NMT     Neural Machine Translation
POS     Parts-Of-Speech
QNLI    Question NLI
RACE    ReAding Comprehension from Examinations
RNN     Recurrent Neural Network
RNN     Recursive Neural Network
RNTN    Recursive Neural Tensor Network
RTE     Recognizing Textual Entailment
SNLI    Stanford Natural Language Inference
SOTA    State-Of-The-Art
SQuAD   Stanford Question Answering Dataset
SRL     Semantic Role Labeling
SST     Stanford Sentiment Treebank
SWAG    Situations With Adversarial Generations
TDNN    Time-Delayed Neural Network
ULMFiT  Universal Language Model Fine-tuning
VAE     Variational Autoenconder
WSD     Word Sense Disambiguation
biLM    Bidirectional Language Model
GPT     Generative Pre-training Transformer
CVT     Cross-View Training
STLR    Slanted Triangular Learning Rates
BPE     Byte Pair Encoding
MLM     Mask Language Model
```




## Metrics

### Perplexity

Perplexity is often used as an intrinsic evaluation metric for gauging how well a language model can capture the real word distribution conditioned on the context.

A <a href="https://en.wikipedia.org/wiki/Perplexity">perplexity</a> of a discrete proability distribution $p$ is defined as the exponentiation of the entropy:

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

[source](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#metric-perplexity)
