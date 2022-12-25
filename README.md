![1_Ph-8B7lWoLjwJ8bYfOkD3Q](https://user-images.githubusercontent.com/109751694/209456502-a62a7d0c-f234-4a9c-8653-7cf2133ddb97.png)

# NER-With-Few-Shot-Learning
NER) is the problem of recognizing and extracting specific types of entities in text. Such as people or place names. In fact, any concrete “thing” that has a name
### Nutrition is one of the crucial topics nowadays, especially Food dissipation is one of the most
### challenging problems around the world, and we believe that the huge effect is caused by
### small changes, so we started with a very simple problem which "lunch dilemma", here we try
### to suggest different ways cook with the same or similar ingredient which will be extracted
### from user text automatically, and that idea can extend to be a really creative food
### recommender for a personal suggestion or commercial one with a restaurant and online food
### apps.
### the system contains different parts chained in a logical flow,
#### 1- The input text: the user's text
#### 2- preprocessing: remove non-interesting parts and standardize data
#### 3- POS & Dependency tree: extract the grammatical tags and logical words dependency.
#### 4- Premise entities extraction: extract the potential entities.
#### 5- Few-shot NER: extract entities using few-shot learning.
#### 6- Cuisine type prediction (classification): predict causing type using extracted ingredients.
#### 7- Suggestion based clustering: suggest different dishes which relate to or are similar to those extracted ingredients.
### We used 2 data sets:
### 1- Indian-food. (Clustering)
#### Data content
##### Indian cuisine consists of a variety of regional and traditional cuisines native to the Indian subcontinent.
#### Data description
##### [name, ingredients, diet, prep_time, cook_time, flavor profile, course, state, region]
### 2- Recipe-ingredients-dataset. (Classification)
#### Data content
#### Different cuisine types with theme ingredients, it collected from different countries around the world.
#### Data description
##### [id, cuisine type, ingredients]
<h3>Data Preparation.</h3>
<h5>1- Remove punctuations
Removing punctuation that adds up noise that brings ambiguity while training the model.</h5>
<h5>2- Remove Digits
Meaningful and meaningless are considered subjective without any sort of reference
where adding digits to word could change the meaning.</h5>
<h5>3- Convert to lowercase
Converting all your data to lowercase helps in the process of preprocessing and in later stages in
the NLP application, when you are doing parsing.</h5>
<h5>4- Remove Stop Words and Stemming
A group of words which are highly frequently used without any additional information,
such as articles, determiners and prepositions called stop-words. By removing this very
commonly used words from text, we can focus on the important words instead. Also
applying stemming on specific language and differ with respect to performance and
accuracy</h5>

<h3>POS and Dependency tree</h3>
In Dependency parsing, various tags represent the relationship between two words in a
sentence. These tags are the dependency tags. Extracting grammatical tags and contextual
dependency

![112](https://user-images.githubusercontent.com/109751694/209456640-eafc2af1-2163-4dc4-937b-f43f53112419.JPG)
<h3>Premise Entities extraction</h3>
the POS tags and Dependency tree results had used here, some Dependency rules have been
defined using grammatical awareness of language and logical dependency, so the Dependency
Matcher of Spacy used here to extract the defied rules, to get all potential entities, and it will be
filtered in the next step to just keep the related entities.
here are some rules we tried to extract:
![image](https://user-images.githubusercontent.com/109751694/209456648-84d96168-92a0-4b64-aa79-23e2757c7ea9.png)
<h3>Few Shot Ner</h3>
Name Entities Extraction NER based on few-shot learning FSL has used here, where you don't
need to train your model for specific entities, which is very helpful for the cases that suffer from
data scarcity.
so instead of input the text to the model and it extracts the entities that have already been
learned, it takes the text with entities definition, and it will try to extract them based on its latent
knowledge representation.
the only challenge here is you need to identify your entities with meaningful words and each
entity can have a few examples that are related to the entity name, and the more informative
words you use, the more accuracy you will get.
So, using mDeBERT model produced by Microsoft, this idea becomes feasible, so it takes text
as input with some potential classes and it produces the association probability for each class
(as SoftMax), it is using its intensive knowledge representation that is built using huge datasets
in 16 different languages following few-shot approach.
So, we try to discriminate which given class is more associated with the given text and using the
soft-classification we can get which entity is more relevant to input text.

![image](https://user-images.githubusercontent.com/109751694/209456659-95ba43b1-3986-4909-ba59-a393efff77d0.png)
![image](https://user-images.githubusercontent.com/109751694/209456662-5c583f2e-bc08-4eb7-90e6-c93923e36cc8.png)

<h3>Cuisine type prediction (Classification)</h3>
Applying feature engineering using: TF-IDF
Using 3 models and getting the Champion model based on the following
the classifier is very conservative - does not risk too much in saying that a sample is Positive
The model maximizes the number of True Positives But it could be wrong sometimes!
high F-score can be the result of an imbalance between Precision and Recall
![image](https://user-images.githubusercontent.com/109751694/209456686-fcead5be-7894-47d4-9273-b0509f8627a1.png)

This Matrix and reportWe tried here to provide an innovative framework based on few-shot approach, the used core
model here was trained on 100 languages and is therefore also suitable for multilingual zeroshot classification, also POS with Dependency Tree (DT) have used to extract the patterns, so
by changing those parts POS & DT to be suitable with other language, the flow will still the
same. our demo here works with Arabic language too, but the output of POS is not good
because its specified for English language only.
The APP is not specified for foods only, it can work in any other areas without changing
anything, which a great start for any project that doesn't has any data done Using YellowBrick Lib
![image](https://user-images.githubusercontent.com/109751694/209456691-fb5c9eed-3ada-4f45-83db-0ef6f570ab4f.png)

<h3>Contribution</h3>
We tried here to provide an innovative framework based on few-shot approach, the used core
model here was trained on 100 languages and is therefore also suitable for multilingual zeroshot classification, also POS with Dependency Tree (DT) have used to extract the patterns, so
by changing those parts POS & DT to be suitable with other language, the flow will still the
same. our demo here works with Arabic language too, but the output of POS is not good
because its specified for English language only.
The APP is not specified for foods only, it can work in any other areas without changing
anything, which a great start for any project that doesn't has any data

<h3>Live Demo</h3>
Using Streamlit provided by Spacy (Streamlit.py)
![image](https://user-images.githubusercontent.com/109751694/209456712-0e3ee952-7a7f-4224-8ac3-9b8c9dbcc2bf.png)

<h3>Limitations</h3>
Limitations the POS and DT that has used here are dedicated for English language only, but the
APP can be customized for other language by changing those parts, also the core model here
mDeBERTa-v3 has really good knowledge representation, but there are other different huge
models can be used for better representation. also increasing number of examples of entities or
number of entities mean long time in inference, it's a trade-off but it can improve, and the
entities should have informative word so the context and meaning can be captured or you got
bad results.
