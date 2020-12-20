<p align="center">
  <img width='650' src='https://github.com/ashleystephenson/APTitude/blob/master/imgs/selex-process.png' alt='SELEX Lifecycle'/>
</p>

# APTitude
This repository contains several working Deep Learning models for tasks related to optimizing SELEX for high-throughput experimentation in a Digital Microfluidic environment.

![Build Status](https://img.shields.io/badge/build-Stable-green.svg)
![License](https://img.shields.io/badge/license-NONE-green.svg)
<br/><br/><br/>

## Contents
* [Prerequisites](https://github.com/ashleystephenson/APTitude/tree/master#prerequisites)
* [Installation](https://github.com/ashleystephenson/APTitude/tree/master#installation)
* [Usage](https://github.com/ashleystephenson/APTitude/tree/master#usage)
* [Authors](https://github.com/ashleystephenson/APTitude/tree/master#authors)
* [Contributing](https://github.com/ashleystephenson/APTitude/tree/master#contributing)
* [Acknowledgments](https://github.com/ashleystephenson/APTitude/tree/master#acknowledgments)
* [License](https://github.com/ashleystephenson/APTitude/tree/master#license)
<br/>

## Prerequisites
  * Python
  * Numpy
  * Matplotlib
<br/><br/>


## Installation
```
  git clone https://github.com/ashleystephenson/APTitude.git
```
<br/>

## Usage
Navigate to the directory of the model you would like to test. Edit the hyperparameters to experiment with finding optimal values for accuracy and quick convergence. The program will:

1. Load the target dataset.
2. Split it into "training," "validation," and "testing" sets.
3. Display or print a random example from the target training set.
4. Train the model, recording the histories for model cost, training and test accuracy, and training duration.
5. Display a plot of the histories recorded from step 4.
6. Display an example from the test set, along with it's classification and label.
7. End.

Feel free to ask us questions on GitHub ([Ashley](https://github.com/ashleystephenson) & [Johnathan](https://github.com/chivington))

<br/>
<p align="center">
  <img width='600' src='https://github.com/ashleystephenson/APTitude/blob/master/out/v1/aptitude-model-cost-plot.png' alt='Classification Test'/>
</p>
<p align="center">
  <img width='600' src='https://github.com/ashleystephenson/APTitude/blob/master/out/v1/aptitude-model-error-plot.png' alt='Classification Test'/>
</p>
<p align="center">
  <img width='600' src='https://github.com/ashleystephenson/APTitude/blob/master/out/v1/aptitude-model-accuracy-plot.png' alt='Classification Test'/>
</p>
<br/><br/>


## Authors
* **Ashley Stephenson:** [GitHub](https://github.com/ashleystephenson)
* **Johnathan Chivington:** [GitHub](https://github.com/chivington)

## Contributing
Not currently accepting outside contributors, but feel free to use as you wish.

## License
There is currently no license associated with this project.
<br/><br/>
