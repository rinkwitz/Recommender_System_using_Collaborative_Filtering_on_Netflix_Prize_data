# Recommender System using Collaborative Filtering on Netflix Prize data

This Project builds a recommender system using collaborative filtering on the Netflix Prize data. 
The implementation uses vectorization in order to compute gradient descent and is done from scratch in Numpy. 

## Prerequisites

The following Python packages are needed:

* [Numpy](https://www.numpy.org/)
* [Pandas](https://pandas.pydata.org/)

For installation use the command:

```
pip3 install numpy pandas
```

## Methodology

### Data

For this project, I used the Netflix Prize data from [Kaggle](https://www.kaggle.com/).

* [Netflix Prize data](https://www.kaggle.com/netflix-inc/netflix-prize-data/downloads/netflix-prize-data.zip/1)

### Preprocessing

The downloaded data ```netflix-prize-data.zip``` has to be extracted to the local working directory of this project. After this run the script ```preprosessing.py``` to create a preprocessed numpy array ```data.npy```. Your machine should have at least 8GB of RAM to preprocess the whole data set.

### Collaborative Filtering

In this collaborative filtering approach we represent the movies using a matrix <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;X\in\mathbb{R}^{n_F\times&space;n_M}" title="X\in\mathbb{R}^{n_F\times n_M}" /> with <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;n_F" title="n_F" /> number of features and <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;n_M" title="n_M" /> number of movies. Each column of the matrix 

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;\boldsymbol{x_i}:=&space;\begin{pmatrix}&space;x_{1i}\\&space;\vdots\\&space;x_{n_Fi}&space;\end{pmatrix}" title="\boldsymbol{x_i}:= \begin{pmatrix} x_{1i}\\ \vdots\\ x_{n_Fi} \end{pmatrix}" />
</p>

describes the features of the <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;i" title="i" />-th movie. A similar representation <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;\Theta\in\mathbb{R}^{n_F\times&space;n_U}" title="\Theta\in\mathbb{R}^{n_F\times n_U}" /> is used for the users with <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;n_U" title="n_U" /> number of users. Again each column of the matrix

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;\boldsymbol{\theta_i}:=&space;\begin{pmatrix}&space;\theta_{1i}\\&space;\vdots\\&space;\theta_{n_Fi}&space;\end{pmatrix}" title="\boldsymbol{\theta_i}:= \begin{pmatrix} \theta_{1i}\\ \vdots\\ \theta_{n_Fi} \end{pmatrix}" />
</p>

describes the features of the <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;j" title="j" />-th user. The different ratings <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;y_{ij}" title="y_{ij}" /> by user <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;j" title="j" /> for the movie <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;i" title="i" />, that ranges from 1 to 5 or is equal to 0 if no rating is provided, are given by the matrix <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;Y\in\mathbb{R}^{n_M\times&space;n_U}" title="Y\in\mathbb{R}^{n_M\times n_U}" />. To indicate, whether a user has provided a rating for a certain movie, we use the matrix <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;R\in\mathbb{R}^{n_M\times&space;n_U}" title="R\in\mathbb{R}^{n_M\times n_U}" /> with elements:   

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;r_{ij}=\begin{cases}&space;1,&space;\text{&space;if&space;user&space;}&space;j\text{&space;has&space;rated&space;movie&space;}&space;i\\&space;0,&space;\text{&space;otherwise&space;}&space;\end{cases}" title="r_{ij}=\begin{cases} 1, \text{ if user } j\text{ has rated movie } i\\ 0, \text{ otherwise } \end{cases}" />
</p>

We can define the cost function as:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;J(\boldsymbol{x}_1,...,\boldsymbol{x}_{n_M},&space;\boldsymbol{\theta}_1,&space;...,\boldsymbol{\theta}_{n_F})&space;=&space;\underbrace{\frac&space;12\sum\limits_{(i,&space;j):r(i,&space;j)=1}\left(\boldsymbol{\theta}_j^\intercal\boldsymbol{x}_i-y_{ij}&space;\right&space;)^2}_{\text{loss}}&space;&plus;\underbrace{\frac\lambda&space;2\sum\limits_{i=1}^{n_M}\sum\limits_{k=1}^{n_F}x_{ki}^2&space;&plus;&space;\frac\lambda&space;2\sum\limits_{j=1}^{n_U}\sum\limits_{k=1}^{n_F}\theta_{kj}^2&space;}_{\text{regularization}}" title="J(\boldsymbol{x}_1,...,\boldsymbol{x}_{n_M}, \boldsymbol{\theta}_1, ...,\boldsymbol{\theta}_{n_F}) = \underbrace{\frac 12\sum\limits_{(i, j):r(i, j)=1}\left(\boldsymbol{\theta}_j^\intercal\boldsymbol{x}_i-y_{ij} \right )^2}_{\text{loss}} +\underbrace{\frac\lambda 2\sum\limits_{i=1}^{n_M}\sum\limits_{k=1}^{n_F}x_{ki}^2 + \frac\lambda 2\sum\limits_{j=1}^{n_U}\sum\limits_{k=1}^{n_F}\theta_{kj}^2 }_{\text{regularization}}" />
</p>

The optimization problem in collaborative filtering can then be stated as:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;\min\limits_{\overset{\boldsymbol{x}_1,&space;...,\boldsymbol{x}_{n_M}}{\boldsymbol{\theta}_1,&space;...,\boldsymbol{\theta}_{n_F}}}J(\boldsymbol{x}_1,&space;...,\boldsymbol{x}_{n_M},&space;\boldsymbol{\theta}_1,&space;...,\boldsymbol{\theta}_{n_F})" title="\min\limits_{\overset{\boldsymbol{x}_1, ...,\boldsymbol{x}_{n_M}}{\boldsymbol{\theta}_1, ...,\boldsymbol{\theta}_{n_F}}}J(\boldsymbol{x}_1, ...,\boldsymbol{x}_{n_M}, \boldsymbol{\theta}_1, ...,\boldsymbol{\theta}_{n_F})" />
</p>

### Optimization

In order to minimize the cost function <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;J" title="J" />, we use
gradient descent in this project:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;x_{ki}:=x_{ki}-\alpha\left(\sum\limits_{j:r(i,&space;j)=1}(\boldsymbol{\theta}_j^\intercal&space;\boldsymbol{x}_i-y_{ij})\theta_{kj}&plus;\lambda&space;x_{ki}&space;\right&space;)" title="x_{ki}:=x_{ki}-\alpha\left(\sum\limits_{j:r(i, j)=1}(\boldsymbol{\theta}_j^\intercal \boldsymbol{x}_i-y_{ij})\theta_{kj}+\lambda x_{ki} \right )" />
</p>
<p align="center">
<img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;\theta_{kj}:=\theta_{kj}-\alpha\left(\sum\limits_{i:r(i,&space;j)=1}(\boldsymbol{\theta}_j^\intercal&space;\boldsymbol{x}_i-y_{ij})x_{ki}&plus;\lambda&space;\theta_{kj}&space;\right&space;)" title="\theta_{kj}:=\theta_{kj}-\alpha\left(\sum\limits_{i:r(i, j)=1}(\boldsymbol{\theta}_j^\intercal \boldsymbol{x}_i-y_{ij})x_{ki}+\lambda \theta_{kj} \right )" />
</p>

### Run the Code

In order to train the recommender system use the script ```collaberative_filtering.py``` or use the interactive juypter notebook ```collaberative_filtering.ipynb```. The matrices created by numpy may exceed your memory. On my laptop I could only train it with batches of <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;3\cdot&space;10^6" title="3\cdot 10^6" /> user ratings.

## Authors

* [Philip Rinkwitz](https://github.com/rinkwitz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgements:

The formulas of this README were create using:
* [Codecogs online Latex editor](https://www.codecogs.com/latex/eqneditor.php)
