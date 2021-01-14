---
layout: post
title:  "Automatic Assessment of Spoken English "
description: This research project was started in september 2020 and will run until June 2021. It represents 50% of the final grade for my Masters in Computer Engineering.   
date:   2020-12-09 21:03:36 
categories: Python Pytorch UNIX 
---

Data augmentation is investigated to improve the performance of automatic grammar error correction using the Transformer model. Notably, using grammatical error generation in native speech corpora and language filtering to create pseudo-speech data. 

## Motivations 

English is the world's most spoken language with over 1.2 billion speakers and this is only predicted to increase. This, combined with the fact that native speakers are vastly outnumbered by non-native speakers, has led to huge growth in the field of computer assisted language learning giving rise to apps such as Duolingo. These tools allow learners to receive reliable and meaningful feedback on their ability e.g their grammar/pronunciation, in an instantaneous and low cost manner. This feedback can then be used by the learner to improve their level. One of the ways in which feedback can be provided to learners on their speech, is through automated grammatical error correction (GEC). GEC in speech is a challenging problem for a variety of reasons. Firstly, we must use an Automatic Speech Recognition (ASR) system to create a transcription from audio, which can introduce transcription errors. In addition, unlike written language, speech includes disfluencies such as hesitations, repetitions and full sentences are not always used. As annotated speech data is limited, the current GEC and Grammatical Error Detection systems are trained on written data for which large corpora exist [Knill][knill]. Although this provides a good model baseline, these models perform significantly poorer than models which have then been fine-tuned/trained entirely on speech data. 
 
 This project will focus on the augmentation of our existing data sources to better improve our models aimed at correcting grammatical errors in non-native spoken English. Thus, several augmentation techniques are considered to provide us with pseudo speech data to improve performance of GEC systems.


## Spoken GEC

Grammatical error correction is the task of correcting a sentence to ensure that it obeys the rules of English grammar. This becomes more difficult when dealing with speech due to disfluencies as mentioned earlier. In spoken GEC a learner's speech is transformed using an ASR system into a written transcript. This is then fed through a deep learning model to produce a grammatically correct sentence which can then be shown to the learner as feedback as shown below. It is important that the model removes or ignores the speech disfluencies such as 's-'.

![texture theme preview](/images/gec-ede.PNG)

## Baseline Spoken GEC System
The system is composed of a deep neural network which in this case is a transformer based on the paper [Attention is all you need][attention]. It is a sequence to sequence deep learning model that uses multihead attention to keep track of long range dependencies and an encoder decoder architecture shown below. Input text is first transformed into vectors known as word embeddings which are then passed into the model. 

![texture theme preview](/images/transformer.png)

Neural networks require a large amount of labelled data for training. However, labelled speech data is difficult to obtain as each sentence must be transcribed and annotated manually by someone trained in grammatical error correction. While a large amount of work has been done to produce corpora for written GEC, the number of spoken corpora is very limited. Consequently, the baseline model is trained on large written corpora and then evaluated on smaller spoken corpora. It is found that the performance achieved is noticeably lower than a model which has been fine-tuned on speech data. Thus, this motivates the use of augmentation techniques to try to create data that more closely matches speech data using the data resources we have.

## Data Augmentation
To investigate the possibilities for data augmentation, we first examine the corpora that we have available. The largest and most abundant corpora are written texts with grammatical error annotations. Additionally, several corpora composed of unlabelled native speech are also available. The former performs poorly due to the differences between speech and written text such as disfluencies. The latter is unusable in its current form as labelled data is required to train the neural networks. 

The two main methods for data augmentation which will be explored in this project are:

* Generation of grammatical errors in transcribed native speech corpora effectively transforming native speech into learner speech. 
* Propagation of disfluencies in labelled learner written corpora to make more speech like. 

Both of these methods aim to produce data equivalent to an annotated learner speech corpus but will still produce samples that would not occur naturally. A key part of this work will reside in the over generation of this augmented data followed by filtering to retain the most in domain data.

### Grammatial Error Generation (GEG)

Generation of grammatical errors has been used before for data augmentation, such as in the work done by Manakul now a PHD student at Cambridge. N-gram error statistics are used to create errors in speech corpora. Reverse neural machine translation is another method, which can be used to create grammatical errors by running the models described previously in reverse. However, as this method has been found to be unreliable in nature, the method in [Chloe et al][kakao] will be implemented here and henceforth referred to as the Kakao method. In this approach we create an error dictionary containing the most common errors found in a reference corpus and propagate them into a native speech corpus. This is similar to the n-gram error statistics approach, however, we also introduce an additional probability of a never before seen error occurring in nouns, verbs and prepositions. Some examples of augmented sentences using the data detailed in section 5.2 are shown below. 

![texture theme preview](/images/geg.PNG)
### Speech Disfluency Propagation

To create speech disfluencies in written corpora we focus on two types of disfluency, repetitions and false starts (the learner starts to respond but then stops and starts a new sentence). We define the maximum number of disfluencies in a sentence and the maximum length of a disfluency. We choose locations at random in the sentence and create a disfluency of a random length. This can occur either by repetition of the preceding words or by using a masked language model (MLM). A MLM is a large pre-trained model designed to predict the words hidden by a 'mask' in a sentence. Thus, by placing a mask token at this position the model will give us something akin to a false start which can be inserted into the original sentence. In this work, the [Roberta MLM][roberta] was used.

### Language Model Filtering

In some cases, it is possible that an error sequence or disfluency could be generated which is highly unlikely to be made by a learner. Therefore, a method of filtering generated sentences to yield a better-matching data set is needed. The approach adopted so far is to train a language model on learner speech which can then measure the similarity between the augmented sentences and learner speech. Filtering is then used to remove highly unlikely phrases. 

The language model that is used here is an n-gram language model which assumes the probability of observing the sentence is given by equation below. Thus, the model gives us a measure of the probability of a sequence given the training data. These models have the advantage that they do not require data with corrections to be trained thus we can use un-annotated learner speech transcriptions. 

![texture theme preview](/images/perplexity.PNG)

To carry out filtering we calculate the perplexity of each sentence using equation above. Low perplexity indicates the probability distribution is good at predicting the sample. We then have two choices for filtering our sentences, use an absolute threshold of perplexity and only accept sentences below this threshold, or look at the perplexity difference between the augmented source and target sentence. In both cases a high value of perplexity indicates a highly unlikely construction. 

## Experiments

To train these large neural networks we require a very high amount of data. Unfortunately, as we need labelled data to train our model such as a corrected version of our transcript, our sources of data are limited – we can’t just pull text from the internet – we need manually annotated learner speech. We do however have a lot more annotated written data as we can see in the table below. So we are going to train our model on the CLC corpus and use the labelled test sets (BULATS, NICT) to evaluate it using a metric called [GLEU][gleu-score] which measures performance of grammatical error systems. 


*	Cambridge Learner Corpus (CLC): range of essay responses to Cambridge Assessments. 
*	NICT-JLE: Manually transcribed oral proficiency tests for Japanese learners.
*	BULATS: [Linguaskill][lingua-skill] business English free speaking tests.


| Corpus         | Spoken/Written &nbsp; &nbsp; &nbsp; | # Words      |
| :--------------| :--------------:| ------------:|
| CLC            | Written         | 14.1M        |
| BULATS         | Spoken          | 61.9K        |
| NICT-JLE       | Spoken          | 135.3k       |

The model that I’m going to be using is a standard Transformer as mentioned in the [Attention is all you need][attention] paper, coded in Pytorch. I am not going to be using pre-trained word embeddings although they would most likely improve results.

## Initial Results

The model was trained for 20 epochs using a GPU and then the best epoch was evaluated on the unseen spoken test sets BULATS and NICT. We can see below the [GLEU scores][gleu-score] for a model trained on the CLC corpus and a model trained on additional data from the 2019 [BEA task][bea]. Despite using almost 50% more data the improvements are almost negligible in the second model: adding more written data will not improve our model.

| Model data     | &nbsp; &nbsp; &nbsp; Data size /tokens  | &nbsp; &nbsp; &nbsp; NICT      | &nbsp; &nbsp; &nbsp; BULATS   |
| :-------------:| ------------------:|---------: |---------:|
| CLC            | 25M                | 0.475     | 0.493    |
| CLC + BEA      | 38.7M              | 0.477     | 0.498    |

## And if we had speech data?

To answer this question we use k-fold cross validation. Fine tuning the CLC and BEA model from before on 80% of the NICT corpus then producing a correction for the other 20%. We do this 5 times then concatenate all the corrections and calculate a GLEU score. We finetune for 3 epochs and then choose the best one. 

![texture theme preview](/images/cross_validation.png)

| Model data     | &nbsp; &nbsp; &nbsp; NICT   |
| :-------------:| ------------------:|
| CLC + BEA              | 0.477                |
| CLC + BEA fine-tuned on NICT            | 0.598                |

We can see a huge improvement in the results. This project will aim to answer the question: can we create augmented data to make up for our lack of speech data. 

```python
print('hi there')
```



```scss
body {
	font-family: 'Nunito Sans', sans-serif;
	line-height: 1.5em;
	margin: 0;
	-webkit-font-smoothing: antialiased;
	-moz-osx-font-smoothing: grayscale;
}
```

[gleu-score]: https://keisuke-sakaguchi.github.io/paper/2015_groundtruth.pdf
[lingua-skill]: https://www.cambridgeenglish.org/exams-and-tests/linguaskill/
[attention]: https://arxiv.org/pdf/1706.03762.pdf
[bea]: https://www.cl.cam.ac.uk/research/nl/bea2019st/
[swbd]: https://catalog.ldc.upenn.edu/LDC97S62
[roberta]: https://arxiv.org/abs/1907.11692
[kakao]: https://www.aclweb.org/anthology/W19-4423/
[knill]: https://ieeexplore.ieee.org/document/8683080