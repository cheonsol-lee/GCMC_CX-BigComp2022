# Multi-relational Stacking Ensemble Recommender System using Cinematic Experience										

- (2022.01.20) [Multi-Relational Stacking Ensemble Recommender System Using Cinematic Experience](https://ieeexplore.ieee.org/document/9736528) in IEEE BigComp2022
- Reference code: [https://github.com/zhoujf620/Motif-based-inductive-GNN-training](https://github.com/zhoujf620/Motif-based-inductive-GNN-training)
- Author's Paper link: [https://arxiv.org/pdf/1706.02263v2.pdf](https://arxiv.org/pdf/1706.02263v2.pdf)
- Reference code: [https://github.com/tobiasweede/rs-via-gnn/tree/main/02_gcmc/movielens](https://github.com/tobiasweede/rs-via-gnn/tree/main/02_gcmc/movielens)


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

- grid_search_final.ipynb


## Results

|Dataset|Our code <br> best of epochs|
|:-:|:-:|
|Rotten Tomatoes|0.8070|
