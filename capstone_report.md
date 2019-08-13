# Machine Learning Engineer Nanodegree
## Capstone Project
Sam Hiatt
Aug 12, 2019

## I. Project Definition

### Overview

#### Background

Many social animals communicate using vocalizations that can give clues to their species as well as to their intent.
[Spectrograms](https://en.wikipedia.org/wiki/Spectrogram) (also called sonograms, visualizations based on sound frequencies) are commonly used for visual representation of audio information and have long been used for studying recordings of animal vocalizations. The project [DeepSqueak](https://github.com/DrCoffey/DeepSqueak) at the University of Washington in Seattle uses spectrograms and takes a deep learning approach to classifying recordings of ultrasonic vocalizations of rodents. Their publication in Nature, [DeepSqueak: a deep learning-based system for detection and analysis of ultrasonic vocalizations](https://www.nature.com/articles/s41386-018-0303-6), uses this classifier to prove correlations between specific behaviors and types of vocalizations.

Here we apply we take a similar approach to attempt to predict bird species from digital audio recordings of their vocalizations. Using recordings from [Xeno Canto](https://www.xeno-canto.org/), a crowd-sourced Creative Commons database of audio recordings from around the world, the [Xeno-Canto Avian Vocalizations CA/NV, USA](https://www.kaggle.com/samhiatt/xenocanto-avian-vocalizations-canv-usa) dataset was curated to jumpstart exploration into this space. This dataset includes 30 varying length audio samples for each of 91 different avian species common in California and Nevada, USA.

### Problem Statement
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_

... ...
* mention the issues with the approach taken to augment samples in the [British Birdsong feature extraction kernel](https://www.kaggle.com/fleanend/extract-features-with-librosa-predict-with-nb). Essentially there aren't enough samples per class to regularize for environmental factors. This approach will not perform well on new data.
* Leave room for reducing the problem space by reducing the number of classes.
....

Using audio data from the [Xeno-Canto Avian Vocalizations CA/NV, USA](https://www.kaggle.com/samhiatt/xenocanto-avian-vocalizations-canv-usa) dataset, this project trains a classifier to predict the most prevalent bird species in a given audio sample of avian vocalizations. [Mel-spectrograms](https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html) and [Mel-frequency Cepstral Coefficients (MFCCs)](https://librosa.github.io/librosa/generated/librosa.feature.mfcc.html), another audio transform commonly used in voice recognition tasks, are used as input features for classification. Several different classification models are implemented, tuned, and evaluated. It is expected that a CNN will be able to achieve improved performance by introducing some temporal invariance to the input data.

This effort focuses on just a small subset of the available data. While many additional samples of each species are available online and could be used to improve the accuracy for any particular species of interest, such refinement is left for future work.

### Metrics
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_

Model performance is evaluated by calculating the [accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html), defined as the number of samples correctly labeled divided by the total number of samples evaluated. Both training and evaluation accuracy are monitored during the model selection and training process, and the resulting model performance is evaluated on a designated test dataset, previously unseen by the model.

## II. Analysis

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _Have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _Are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

...  
* Show distributions of classes.
  * Recordings per class
  * seconds per class.
* Show some spectrograms and MFCCs.
* Plot some histograms of spectrogram pixels to show their exponential distribution.
...  

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

* describe use of spectrograms, time on one axis, frequency on the other

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_

A purely random predictor would be correct 1.1% of the time (1/91 classes). A [Gaussian Naive Bayes classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) should perform better than random guessing and is used as a benchmark predictor. The naive assumption of feature independence is expected to limit the accuracy of this predictor. It is expected to be able to pick up on certain frequency bands that are common for a particular species' vocalizations.

## III. Methodology

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_


## References

[Bird Song Research: The Past 100 Years](https://courses.washington.edu/ccab/Baker%20-%20100%20yrs%20of%20birdsong%20research%20-%20BB%202001.pdf)

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
