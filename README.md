![X](logo.jpg)

# Polytechnique [CSE204-2019](https://moodle.polytechnique.fr/course/view.php?id=7862)

These courses were made by [Jérémie Decock](http://www.jdhp.org/) ([lab sessions 01 to 04](https://github.com/jeremiedecock/polytechnique-cse204-2018)), [Théo Lacombe](https://tlacombe.github.io/) (lab exam 05) and [James B. Scoggins](https://jbscoggi.github.io/website/) ([lab sessions 06 to 13](https://github.com/jbscoggi/teaching/tree/master/Polytechnique/CSE204)) for [CSE204-2018](https://moodle.polytechnique.fr/enrol/index.php?id=6784), slightly amended and put together by [Pawel Guzewicz](http://www.lix.polytechnique.fr/Labo/Pawel.Guzewicz/) and I ([Adrien Ehrhardt](https://adimajo.github.io/)).

## Moodle

[Moodle Link](https://moodle.polytechnique.fr/course/view.php?id=7862)

### Welcome to CSE204 Machine Learning!

Machine learning is an increasingly important area, and it has provided many of the recent advances behind applications of artificial intelligence. It is relevant to a plethora of application domains in science and industry - including in finance, health, transport, linguistics, media, and biology. Lectures will cover the many of the main concepts and algorithms. We will cover in some degree all the main paradigms of machine learning: supervised learning (regression, classification), unsupervised learning, and reinforcement learning. Among many learning algorithms we will look at: least squares, logistic regression, k-nearest neighbors, neural networks and deep learning, decision tree inducers, kernel methods, PCA, k-means clustering, and Q-learning. In the labs, we will implement many of these, and investigate their use in different applications. Programming will be done in Python with scientific libraries such as numpy and scikit-learn. The main grading component is a team project, as well as several in-class tests.

**Course Outline**: A working outline can be found as topics below. This is subject to minor changes as the course progresses. 


**Grading:**

2 Lab reports/in-class tests 25% each

Group project (in groups of 3) 50%

**Some recommended literature:**

There is no official course textbook, but the following are recommendations (others may be added later):

James et al., **An Introduction to Statistical Learning**. Springer. (And/or: Hastie et al., The Elements of Statistical Learning. Springer)

Goodfellow et al. **The Deep Learning Book**. MIT Press.

**Lab sessions:**

Jupyter Notebooks can either be executed on Google Colab (works nicely with Google Drive), MyBinder (don't forget to regularly save your work) or locally, in which case Anaconda (2019.10 or above ; Python 3.7 or above) is strongly recommended.
If the conda command does not work, it's that conda is not in the PATH environment variable. You may add it with the command:

`export PATH="/usr/local/Anaconda3-2019.10/bin:$PATH"`

`conda init`

We use `nbgrader` extensively: cells in which you have to input your own code are clearly shown; other cells are read-only. You may test your answers with the "Validate" button of the "Assignment list" in Jupyter Notebook. Warning: this is only available through a local installation and requires the installation of nbgrader using:

`conda install -c conda-forge nbgrader`

`jupyter nbextension install --user --py nbgrader --overwrite`

`jupyter nbextension enable --user --py nbgrader`

Although all notebooks rely on nbgrader, only sessions 05 and 12 are graded.
If you choose the "Local" version of the lab sessions, right click on the link > Save Link as...
If you wish to execute the notebooks on the machines of the Salle d'Informatique, download [this conda_environment.yml](conda_environment.yml) file open a terminal prompt and enter:

`conda env create -f conda_environment.yml`

`python -m ipykernel install --user --name cse204 --display-name "Python (cse204)"`

This will install all dependencies in a new conda environment named cse204. Once on the Notebook, don't forget to use Kernel > Change Kernel to use the Python (cse204) environment.
If the conda environment cse204 already exists, you may delete it with the following command:

`conda remove --name cse204 --all`

To launch Jupyter, use:

`jupyter-notebook --ip=0.0.0.0 --port=8080`

## Lab sessions

## Lab session 01: Introduction

### Subject:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adimajo/polytechnique-cse204-2019-releases/blob/master/lab_session_01/lab_session_01.ipynb)

[![My Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adimajo/polytechnique-cse204-2019-releases/master?filepath=lab_session_01%2Flab_session_01.ipynb)

[![Local](https://img.shields.io/badge/Local-Save%20As...-blue)](https://github.com/adimajo/polytechnique-cse204-2019-releases/raw/master/lab_session_01/lab_session_01.ipynb)

[![Solutions](https://img.shields.io/badge/Solution-As%20HTML-blueviolet)](https://htmlpreview.github.io/?https://github.com/adimajo/polytechnique-cse204-2019-releases/blob/master/lab_session_01/lab_session_01_solutions.html)

## Lab session 02: parametric models

### Subject:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adimajo/polytechnique-cse204-2019-releases/blob/master/lab_session_02/lab_session_02.ipynb)

[![My Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adimajo/polytechnique-cse204-2019-releases/master?filepath=lab_session_02%2Flab_session_02.ipynb)

[![Local](https://img.shields.io/badge/Local-Save%20As...-blue)](https://github.com/adimajo/polytechnique-cse204-2019-releases/raw/master/lab_session_02/lab_session_02.ipynb)

## Lab session 03: `k`-nearest neighbors

### Subject:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adimajo/polytechnique-cse204-2019-releases/blob/master/lab_session_03/lab_session_03.ipynb)

[![My Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adimajo/polytechnique-cse204-2019-releases/master?filepath=lab_session_03%2Flab_session_03.ipynb)

[![Local](https://img.shields.io/badge/Local-Save%20As...-blue)](https://github.com/adimajo/polytechnique-cse204-2019-releases/raw/master/lab_session_03/lab_session_03.ipynb)

## Lab session 04: regression methods

### Subject:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adimajo/polytechnique-cse204-2019-releases/blob/master/lab_session_04/lab_session_04.ipynb)

[![My Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adimajo/polytechnique-cse204-2019-releases/master?filepath=lab_session_04%2Flab_session_04.ipynb)

[![Local](https://img.shields.io/badge/Local-Save%20As...-blue)](https://github.com/adimajo/polytechnique-cse204-2019-releases/raw/master/lab_session_04/lab_session_04.ipynb)

## Lab session 05: Exam 1

## Lab session 06: building a neural network from scratch (part 1)

### Subject:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adimajo/polytechnique-cse204-2019-releases/blob/master/lab_session_06_07/lab_session_06_07.ipynb)

[![My Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adimajo/polytechnique-cse204-2019-releases/master?filepath=lab_session_06_07%2Flab_session_06_07.ipynb)

[![Local](https://img.shields.io/badge/Local-Save%20As...-blue)](https://github.com/adimajo/polytechnique-cse204-2019-releases/raw/master/lab_session_06/lab_session_06.ipynb)


## Lab session 07: building a neural network from scratch (part 2)

See Session 06.

## Lab session 08: Classification of the CIFAR-10 dataset using CNNs

### Subject:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adimajo/polytechnique-cse204-2019-releases/blob/master/lab_session_08/lab_session_08.ipynb)

[![My Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adimajo/polytechnique-cse204-2019-releases/master?filepath=lab_session_08%2Flab_session_08.ipynb)

[![Local](https://img.shields.io/badge/Local-Save%20As...-blue)](https://github.com/adimajo/polytechnique-cse204-2019-releases/raw/master/lab_session_08/lab_session_08.ipynb)


## Lab session 09: Decision Trees and Ensemble Methods

### Subject:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adimajo/polytechnique-cse204-2019-releases/blob/master/lab_session_09/lab_session_09.ipynb)

[![My Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adimajo/polytechnique-cse204-2019-releases/master?filepath=lab_session_09%2Flab_session_09.ipynb)

[![Local](https://img.shields.io/badge/Local-Save%20As...-blue)](https://github.com/adimajo/polytechnique-cse204-2019-releases/raw/master/lab_session_09/lab_session_09.ipynb)


## Lab session 10: Dimensionality Reduction with PCA and Autoencoders

### Subject:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adimajo/polytechnique-cse204-2019-releases/blob/master/lab_session_10/lab_session_10.ipynb)

[![My Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adimajo/polytechnique-cse204-2019-releases/master?filepath=lab_session_10%2Flab_session_10.ipynb)

[![Local](https://img.shields.io/badge/Local-Save%20As...-blue)](https://github.com/adimajo/polytechnique-cse204-2019-releases/raw/master/lab_session_10/lab_session_10.ipynb)


## Lab session 11: Unsupervised Learning - Clustering

### Subject:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adimajo/polytechnique-cse204-2019-releases/blob/master/lab_session_11/lab_session_11.ipynb)

[![My Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adimajo/polytechnique-cse204-2019-releases/master?filepath=lab_session_11%2Flab_session_11.ipynb)

[![Local](https://img.shields.io/badge/Local-Save%20As...-blue)](https://github.com/adimajo/polytechnique-cse204-2019-releases/raw/master/lab_session_11/lab_session_11.ipynb)


## Lab session 12: Exam 2

## Lab session 13: Reinforcement Learning

### Subject:

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adimajo/polytechnique-cse204-2019-releases/blob/master/lab_session_13/lab_session_13.ipynb)

[![My Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adimajo/polytechnique-cse204-2019-releases/master?filepath=lab_session_13%2Flab_session_13.ipynb)

[![Local](https://img.shields.io/badge/Local-Save%20As...-blue)](https://github.com/adimajo/polytechnique-cse204-2019-releases/raw/master/lab_session_13/lab_session_13.ipynb)
