# Machine Learning Engineer Nanodegree
## Capstone Proposal
Sam Hiatt
Aug 5, 2019

### Background
_(approx. 1-2 paragraphs)_

**_In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required._**

Many social animals communicate using vocalizations that can give clues to their species as well as to their intent. Studies of animal vocalizations have been hindered by the cost required for expert analysis of audio signals. Machine learning offers tools than can automate classification of these audible signals, opening up countless opportunities for sound-aware computer applications and could help accelerate studies of these animals. For example, imagine a computer trained to recognize the call of a specific species of bird. This model could be used to trigger a camera recording, or automatically tag a live audio stream containing avian calls with the species of the bird that made it, producing a timeseries record of the presence of this species.

The project [DeepSqueak](https://github.com/DrCoffey/DeepSqueak) at the University of Washington in Seattle takes a deep learning approach classifying ultrasonic recordings of rodents and is able to show correlations between behavior and types of vocalizations. Their publication in Nature, [DeepSqueak: a deep learning-based system for detection and analysis of ultrasonic vocalizations](https://www.nature.com/articles/s41386-018-0303-6), shows how CNNs for image analysis can be used for classifying digital audio signals. 


### Problem Statement
_(approx. 1 paragraph)_

**_In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms), measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once)._**

This project will build a classifier to detect bird species and call type in audio recordings. It should take an audio clip as input and return a label representing the most prevalent / most likely species of bird in the clip, or the most prevalent / most likely call type. 


### Datasets and Inputs
_(approx. 2-3 paragraphs)_

**_In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem._**

[Xeno-Canto](https://www.xeno-canto.org) is an online community and Creative Commons database of crowd-sourced recordings of birds from around the world, indexed by species and labeled by call type. The [British Birdsong Dataset](https://www.kaggle.com/rtatman/british-birdsong-dataset), available as a Kaggle dataset, is a small subset of this database and contains 264 recordings from 88 species commonly heard in the United Kingdom and includes a balanced number of training samples per class and an imbalanced test dataset. 

The study [Individual recognition of opposite sex vocalizations in the zebra finch](https://www.nature.com/articles/s41598-017-05982-x) by researchers at the Max Planck Institute for Ornithology in Germany showed that zebra finches can recognize their mates' vocalizations with a purely audible stimulus. While this study focused on individual recognition, it also produced a [high-quality, publically available dataset](https://datadryad.org/resource/doi:10.5061/dryad.4g8b7/1) of individual zebra finch calls labeled by bird id and call type. 


### Solution Statement
_(approx. 1 paragraph)_

**_In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms), measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once)._**

Drawing on insights from the project DeepSqueak, this effort will attempt to improve upon existing machine learning models for avian vocalization classification by applying a CNN architecture for detecting species and call type in an audio recording. It is expected that the temporal invariance introduced through convolution and max pooling will achieve improved results compared to models that do not preserve the temporal relation between input features.


### Benchmark Model
_(approximately 1-2 paragraphs)_

**_In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail._**

The [Kaggle kernel by Edoardo Ferrante](https://www.kaggle.com/fleanend/extract-features-with-librosa-predict-with-nb) uses the same British Birdsong dataset and creates sonograms for each audio sample. It includes a benchmark species classification model showing a Naive Bayes classifier achieving 86% test accuracy. [Another Kaggle kernel by Edoardo Ferrante](https://www.kaggle.com/fleanend/bird-visualisation-and-classification) uses the same sonograms from the previous kernel and implements classifiers with improved accuracy, 97.7% by using a Multi-layer Perceptron model, and 98.3% by using a random forest model. 

There are no known call type classifiers based on the Zebra Finch dataset so a new benchmark will be needed. Since a Random Forest classifier was the best benchmark for the British Birdsong dataset, a Random Forest model will similarly be trained on sonograms to be used as a benchmark model for the call type classifier. 

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

**_In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms)._**

The accuracy of each classification model will be evaluated using cross-validation, computing training accuracy and validation accuracy, defined as the number of correctly labeled samples divided by the total number of samples in each fold. Test accuracy for models trained on the British Birdsong dataset will be evaluated using the provided test dataset. Evaluation of test accuracy for the Zebra Finch call type classifier will use a subset designated from the beginning to be used as a test set. 


### Project Design
_(approx. 1 page)_

**_In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project._**


* create sonograms and visualize
* attempt to reproduce British Birdsong results
* train a model with a CNN and compare results
* implement the same workflow on the Zebra Finch dataset
* compare performance of Random Forest classifier vs CNN


## References
* Kevin R. Coffey, Russell G. Marx & John F. Neumaier (2019) [DeepSqueak: a deep learning-based system for detection and analysis of ultrasonic vocalizations](https://doi.org/10.1038/s41598-017-05982-x) Neuropsychopharmacology vol 44, pp 859–868. 
* D’Amelio PB, Klumb M, Adreani MN, Gahr ML, ter Maat A (2017) [Individual recognition of opposite sex vocalizations in the zebra finch](https://doi.org/10.1038/s41598-017-05982-x). Scientific Reports 7(1): 5579. 
* D'Amelio PB, Klumb M, Adreani MN, Gahr M, ter Maat A (2017) Data from: [Individual recognition of opposite sex vocalizations in the zebra finch](https://doi.org/10.5061/dryad.4g8b7). Dryad Digital Repository. 
* [Xeno-Canto](https://www.xeno-canto.org), founded by Bob Planqué and Willem-Pier Vellinga. 

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
