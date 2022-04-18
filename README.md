# Multi-relational Stacking Ensemble Recommender System using Cinematic Experience										
	Last modified in 2021.10.20
- (2021.10.15) Cheonsol Lee submitted short paper in [IEEE BigComp2022](http://www.bigcomputing.org/), [Code](https://github.com/cheonsol-lee/bigcomp_2022_multi_graph)
- Reference code: [https://github.com/zhoujf620/Motif-based-inductive-GNN-training](https://github.com/zhoujf620/Motif-based-inductive-GNN-training)
- Author's Paper link: [https://arxiv.org/pdf/1706.02263v2.pdf](https://arxiv.org/pdf/1706.02263v2.pdf)
- Reference code: [https://github.com/zhoujf620/Motif-based-inductive-GNN-training](https://github.com/zhoujf620/Motif-based-inductive-GNN-training)


## Dependencies

* PyTorch 1.2+
* DGL 0.5 (nightly version)

## Data

Movie data
- Rotten Tomatoes data : [https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset](https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)

Training set for BERT
- Sentiment Analysis : [https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)
- Emotion Analysis : [https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp](https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp)


## How to run

- explicit_train_rotten.ipynb


## Results

|Dataset|Our code <br> best of epochs|
|:-:|:-:|
|Rotten Tomatoes|0.8004|
|Amazon Movie|0.9621|
