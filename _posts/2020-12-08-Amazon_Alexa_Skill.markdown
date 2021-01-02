---
layout: post
title:  "Amazon Alexa Skill "
description: This research project was started in september 2020 and will run until June 2021. It represents 50% of the final grade for my Masters in Computer Engineering.   
date:   2020-12-09 21:03:36 
categories: Python Pytorch UNIX 
---

Data augmentation is investigated to improve the performance of automatic grammar error correction using the Transformer model. Notably, using grammatical error generation in native speech corpora and language filtering to create pseudo-speech data. 

## Motivations 

English is the most spoken language in the world with over 1.2 billion speakers and this is only predicted to increase. This combined with the fact that native speakers are vastly outnumbered by non-native speakers has led to huge growth in the field of automatic assessment. One of the ways we would like to do this is by providing learners feedback on their grammar whilst speaking. This has been done for a long time for writing (e.g grammarly) but when it comes to speech this is a challenging problem.

Even native speakers often don’t speak in full sentences, hesitate and repeat themselves. In free speech we also have disfluencies which disrupt the language flow but aren’t necessarily errors. 


## The process
![texture theme preview](/Images/asr.PNG)

Firstly, we use and automated speech recognition system (ASR) to transform our speech into a transcript. It’s worth noting that this can also introduce additional transcription errors. Then we pipe this text into a sequence to sequence deep learning model (e.g RNN, LSTM, transformer) to produce a corrected transcript which we can provide to the learner as feedback. Think of it like neural-machine translation, except we are translating from incorrect learner speech to grammatically correct speech. The text has to be first transformed so that each word is assigned a vector called a word embedding before being fed into the model. 

![texture theme preview](/Images/RNN.PNG)

## The model

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

![texture theme preview](/Images/cross_validation.png)

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