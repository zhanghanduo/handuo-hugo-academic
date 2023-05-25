---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Seq2seq Model with Attention"
subtitle: ""
summary: ""
authors: ["Handuo"]
tags: ["ML", "Notes"]
categories: ["Deep Learning Basics"]
date: 2020-06-01T12:58:12+08:00
lastmod: 2020-06-01T12:58:12+08:00
featured: true
draft: false
markup: mmark

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: "Smart"
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
> Digested and reproduced from <a href="https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Visualizing A Neural Machine Translation Model</a> by Jay Alammar.

{{% toc %}}

Sequence-to-sequence models are deep learning models that have achieved a lot of success in tasks like machine translation, text summarization, and image captioning. [Google Translate](https://blog.google/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/) started  using such a model in production in late 2016. These models are explained in the two pioneering papers ([Sutskever et al., 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), [Cho et al., 2014](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf)).

A sequence-to-sequence model is a model that takes a sequence of items (words, letters, features of an images...etc) and outputs another sequence of items.

In neural machine translation, a sequence is a series of words, processed one after another. The output is, likewise, a series of words:

{{< video src="seq2seq_2.mp4" controls="yes" autoplay="true" loop="true">}}


## 1. Architecture of Seq2seq

The model is composed of an {{< hl >}}encoder{{< /hl >}} and a {{< hl >}}decoder{{< /hl >}}. The {{< hl >}}encoder{{< /hl >}} processes each item in the input sequence and compiles the information into a vector (called the {{< hl >}}context{{< /hl >}} ). After processing the input sequence, the {{< hl >}}encoder{{< /hl >}} sends the {{< hl >}}context{{< /hl >}} over to the {{< hl >}}decoder{{< /hl >}} , which produces the output sequence item by item.

{{< video src="seq2seq_3.mp4" controls="yes" autoplay="true" loop="true">}}


The same applies in the case of machine translation.

{{< video src="seq2seq_4.mp4" controls="yes" autoplay="true" loop="true">}}


RNN (recurrent neural network) is shown as an illustration here to be the model of both {{< hl >}}encoder{{< /hl >}} and {{< hl >}}decoder{{< /hl >}} (Be sure to check out Luis Serrano's [A friendly introduction to Recurrent Neural Networks](https://www.youtube.com/watch?v=UNmqTiOnRfg) for an intro to RNNs).

{{< figure src="context.png" title="The **context** is a vector of floats. Later in this post we will visualize vectors in color by assigning brighter colors to the cells with higher values." lightbox="true" >}}

You can set the size of the {{< hl >}}context{{< /hl >}} vector when you set up your model. It is basically the number of hidden units in the {{< hl >}}encoder{{< /hl >}} RNN. These visualizations show a vector of size 4, but in real world applications the {{< hl >}}context{{< /hl >}} vector would be of a size like 256, 512, or 1024.

By design, a RNN takes two inputs at each time step: an input (in the case of the encoder, one word from the input sentence), and a hidden state. The word, however, needs to be represented by a vector. To transform a word into a vector, we turn to the class of methods called "[word embedding](https://machinelearningmastery.com/what-are-word-embeddings/)" algorithms. These turn words into vector spaces that capture a lot of the meaning/semantic information of the words (e.g. [king - man + woman = queen](http://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html)).

{{< figure src="embedding.png" title="We need to turn the input words into vectors before processing them. That transformation is done using a **word embedding** algorithm. We can use [pre-trained embeddings](http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/) or train our own embedding on our dataset. Embedding vectors of size 200 or 300 are typical, we're showing a vector of size four for simplicity."  lightbox="true" >}}

Now that we've introduced our main vectors/tensors, let's recap the mechanics of an RNN and establish a visual language to describe these models:

{{< video src="RNN_1.mp4" controls="yes" autoplay="true" loop="true">}}

The next RNN step takes the second input vector and hidden state #1 to create the output of that time step. Later in the post, we'll use an animation like this to describe the vectors inside a neural machine translation model.

In the following visualization, each pulse for the {{< hl >}}encoder{{< /hl >}} or {{< hl >}}decoder{{< /hl >}}  is that RNN processing its inputs and generating an output for that time step. Since the {{< hl >}}encoder{{< /hl >}} and {{< hl >}}decoder{{< /hl >}} are both RNNs, each time step one of the RNNs does some processing, it updates its {{< hl >}}hidden state{{< /hl >}}  based on its inputs and previous inputs it has seen.

Let's look at the {{< hl >}}hidden states{{< /hl >}} for the {{< hl >}}encoder{{< /hl >}}. Notice how the last {{< hl >}}hidden state{{< /hl >}}  is actually the {{< hl >}}context{{< /hl >}}  we pass along to the {{< hl >}}decoder{{< /hl >}}.

{{< video src="seq2seq_5.mp4" controls="yes"  autoplay="true" loop="true">}}

The {{< hl >}}decoder{{< /hl >}} also maintains a {{< hl >}}hidden states{{< /hl >}} that it passes from one time step to the next. We just didn't visualize it in this graphic because we're concerned with the major parts of the model for now.

Let's now look at another way to visualize a sequence-to-sequence model. This animation will make it easier to understand the static graphics that describe these models. This is called an "unrolled" view where instead of showing the one {{< hl >}}decoder{{< /hl >}}, we show a copy of it for each time step. This way we can look at the inputs and outputs of each time step.

{{< video src="seq2seq_6.mp4" controls="yes"  autoplay="true" loop="true">}}

## 2. Attention Mechanism

Attention is a generalized pooling method with. The core component in the attention mechanism is the attention layer, or called attention for simplicity. An input of the attention layer is called a query. For a query, attention returns an o bias alignment over inputsutput based on the memory — a set of key-value pairs encoded in the attention layer. To be more specific, assume that the memory contains $n$ key-value pairs, $(k_1, v_1), \cdots, (k_n, v_n)$ with $\mathbf{k}\in{ \mathbb{R}^{d_k}}, \mathbf{v}\in{ \mathbb{R}^{d_v}}$. Given a query $\mathbf{q}\in \mathbb{R}^{d_q}$, the attention layer returns an output $\mathbf{o}\in{ \mathbb{R}^{d_v}} $ with the same shape as the value. 



Pros of using attention:

1. With attention, Seq2seq does not forget the source input.
2. With attention, the decoder knows where to focus.

The {{< hl >}}context{{< /hl >}} vector turned out to be a bottleneck for these types of models. It made it challenging for the models to deal with long sentences. A solution was proposed in [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) and [Luong et al., 2015](https://arxiv.org/abs/1508.04025). These papers introduced and refined a technique called "Attention", which highly improved the quality of machine translation systems. Attention allows the model to focus on the relevant parts of the input sequence as needed.

{{< figure src="attention.png" title="At time step 7, the attention mechanism enables the **decoder** to focus on the word **étudiant** (*student* in french) before it generates the English translation. This ability to amplify the signal from the relevant part of the input sequence makes attention models produce better results than models without attention."  lightbox="true" >}}

Let's continue looking at attention models at this high level of abstraction. An attention model differs from a classic sequence-to-sequence model in two main ways:

First, the encoder passes a lot more data to the decoder. Instead of passing the last hidden state of the encoding stage, the encoder passes _all_ the hidden states to the decoder:

{{< video src="seq2seq_7.mp4" controls="yes"  autoplay="true" loop="true">}}

Second, an attention decoder does an extra step before producing its output. In order to focus on the parts of the input that are relevant to this decoding time step, the decoder does the following:

 1. Look at the set of encoder {{< hl >}}hidden states{{< /hl >}} it received -- each encoder hidden states  is most associated with a certain word in the input sentence
 2. Give each {{< hl >}}hidden states{{< /hl >}} a score (let's ignore how the scoring is done for now)
 3. Multiply each {{< hl >}}hidden states{{< /hl >}} by its softmaxed score, thus amplifying {{< hl >}}hidden states{{< /hl >}} with high scores, and drowning out {{< hl >}}hidden states{{< /hl >}} with low scores

{{< video src="attention_process.mp4" controls="yes"  autoplay="true" loop="true">}}

This scoring exercise is done at each time step on the {{< hl >}}decoder{{< /hl >}} side.

Let us now bring the whole thing together in the following visualization and look at how the attention process works:

1. The attention decoder RNN takes in the embedding of the \<END\> token, and an initial decoder hidden state.   
2. The RNN processes its inputs, producing an output and a new hidden state vector (h4). The output is discarded.   
3. Attention Step: We use the encoder hidden states and the h4 vector to calculate a context vector (C4) for this time step.
4. We concatenate h4 and C4 into one vector.
5. We pass this vector through a {{< hl >}}feedforward neural network{{< /hl >}} (one trained jointly with the model).
6. The output of the feedforward neural networks indicates the output word of this time step.
7. Repeat for the next time steps

{{< video src="attention_tensor_dance.mp4" controls="yes"  autoplay="true" loop="true">}}

This is another way to look at which part of the input sentence we're paying attention to at each decoding step:

{{< video src="seq2seq_9.mp4" controls="yes"  autoplay="true" loop="true">}}

Note that the model isn't just mindless aligning the first word at the output with the first word from the input. It actually learned from the training phase how to align words in that language pair (French and English in our example). An example for how precise this mechanism can be comes from the attention papers listed above:

{{< figure src="attention_sentence.png" title="You can see how the model paid attention correctly when outputing **European Economic Area**. In French, the order of these words is reversed (*européenne économique zone*) as compared to English. Every other word in the sentence is in similar order."  lightbox="true" >}}

You can check TensorFlow's [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt).