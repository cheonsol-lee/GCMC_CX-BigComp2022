"""RottenMovie dataset"""
import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
import torch as th

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir
from utils import to_etype_name
import pandas as pd
import os
import datetime
import numpy as np
import pandas as pd

class RottenMovie(object):
    def __init__(self,
                 # paths for dataset
                 train_data,
                 test_data,
                 movie_data,
                 user_data,

                 emotion=False,
                 sentiment=False,

                 name='rotten', 
                 device='cpu',
                 mix_cpu_gpu=False,
                 use_one_hot_fea=False, 
                 symm=True,
                 valid_ratio=0.1,
                 seed=123
                 ):

        self._name = name
        self._device = device
        self._symm = symm
        self._valid_ratio = valid_ratio
        self.sentiment = sentiment
        self.emotion = emotion

        print('......1: 데이터 로드')
        self.all_train_rating_info = pd.read_csv(train_data, encoding='utf-8')
        self.test_rating_info = pd.read_csv(test_data, encoding='utf-8')
        
        #######
        if self.sentiment == True:
            self.all_train_rating_info['sentiment'] = self.all_train_rating_info['sentiment'] + 11 #rating 과 구분하기위해 11 더해줌 : 11 - 15
            self.test_rating_info['sentiment'] = self.test_rating_info['sentiment'] + 11 #rating 과 구분하기위해 11 더해줌 : 11 - 15

        if self.emotion == True:
            self.all_train_rating_info['emotion'] = self.all_train_rating_info['emotion'] + 16 # 16~21
            self.test_rating_info['emotion'] = self.test_rating_info['emotion'] + 16 # 16~21

        self.all_rating_info = pd.concat([self.all_train_rating_info, self.test_rating_info], ignore_index=True)

        self.user_info = pd.read_csv(user_data, encoding='utf-8')

        #movie data processing
        movie_df = pd.read_csv(movie_data, encoding='utf-8')
        movie_df.reset_index(inplace=True)
        movie_df.rename(columns = {"index": "movie_id"}, inplace=True)

        movie_df = movie_df[['movie_id','movie_title','original_release_date', 'genres']]
        movie_df['original_release_date']= pd.to_datetime(movie_df['original_release_date']) 
        movie_df['year'] = movie_df['original_release_date'].dt.strftime('%Y')
        mean_val = pd.to_numeric(movie_df['year']).mean()

        movie_df = movie_df.fillna({'original_release_date':movie_df.original_release_date.mean()})
        movie_df = movie_df.fillna({'year':mean_val})
        movie_df = movie_df.fillna({'genres' : 'unknown'})

        self.movie_info = movie_df
        self.movie_genre_dummies = movie_df['genres'].str.get_dummies(sep=',')
        self.movie_info = pd.concat([movie_df, self.movie_genre_dummies], axis=1)
        
        
        print('......3: Train/Valid 분리')
        np.random.seed(seed)
        th.manual_seed(seed)
        dgl.seed(seed)
        print(f"seed:{seed}")
        
        num_valid = int(np.ceil(self.all_train_rating_info.shape[0] * valid_ratio))
        shuffled_idx = np.random.permutation(self.all_train_rating_info.shape[0])
        self.train_idx = shuffled_idx[num_valid: ]
        self.valid_idx = shuffled_idx[ :num_valid]
        self.train_rating_info = self.all_train_rating_info.iloc[self.train_idx]
        self.valid_rating_info = self.all_train_rating_info.iloc[self.valid_idx]


        print("All rating pairs : {}".format(self.all_rating_info.shape[0]))
        print("\tAll train rating pairs : {}".format(self.all_train_rating_info.shape[0]))
        print("\t\tTrain rating pairs : {}".format(self.train_rating_info.shape[0]))
        print("\t\tValid rating pairs : {}".format(self.valid_rating_info.shape[0]))
        print("\tTest rating pairs  : {}".format(self.test_rating_info.shape[0]))
        

        print('......4: User/Movie를 Global id에 매핑')
        # Map user/movie to the global id
        self.global_user_id_map = {ele: i for i, ele in enumerate(self.user_info['user_id'])}
        self.global_movie_id_map = {ele: i for i, ele in enumerate(self.movie_info['movie_id'])}
        print('Total user number = {}, movie number = {}'.format(len(self.global_user_id_map),
                                                                 len(self.global_movie_id_map)))
        self._num_user = len(self.global_user_id_map)
        self._num_movie = len(self.global_movie_id_map)

        
        print('......5: features 생성')
        ### Generate features

        self.user_feature = None
        self.movie_feature = None

        # load feature
        if use_one_hot_fea == False:
            self.user_feature = th.FloatTensor(self._process_user_fea())
            self.movie_feature = th.FloatTensor(self._process_movie_fea())
            
            # if mix_cpu_gpu, we put features in CPU
            if mix_cpu_gpu == False:
                self.user_feature = self.user_feature.to(self._device)
                self.movie_feature = self.movie_feature.to(self._device)
                
        if self.user_feature is None:
            self.user_feature_shape = (self.num_user, self.num_user)
            self.movie_feature_shape = (self.num_movie, self.num_movie)
        else:
            self.user_feature_shape = self.user_feature.shape
            self.movie_feature_shape = self.movie_feature.shape
            
        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nmovie: {}".format(self.movie_feature_shape)
        print(info_line)

        
        print('......6: Graph Encoder/Decoder 생성')

        self.emotion_rating_values = list(set(self.all_rating_info["emotion"].values))
        self.sentiment_rating_values = list(set(self.all_rating_info["sentiment"].values))
        self.possible_rating_values = list(set(self.all_rating_info["rating"].values))
        
        self.rating_values = self.possible_rating_values
        if self.sentiment == True:
            self.rating_values += self.sentiment_rating_values
        if self.emotion == True:
            self.rating_values += self.emotion_rating_values

        print("rating_values : ", self.rating_values)
        all_rating_pairs, all_rating_values, all_sentiment_values, all_emotion_values = self._generate_pair_value(self.all_rating_info)
        all_train_rating_pairs, all_train_rating_values, all_train_sentiment_values, all_train_emotion_values = self._generate_pair_value(self.all_train_rating_info)
        train_rating_pairs, train_rating_values, train_sentiment_values, train_emotion_values = self._generate_pair_value(self.train_rating_info)
        valid_rating_pairs, valid_rating_values, valid_sentiment_values, valid_emotion_values = self._generate_pair_value(self.valid_rating_info)
        test_rating_pairs, test_rating_values, test_sentiment_values, test_emotion_values = self._generate_pair_value(self.test_rating_info)

        self.train_s_graph, self.train_e_graph = self._generate_sub_graph(train_rating_pairs, train_sentiment_values, train_emotion_values)
        self.train_enc_graph = self._generate_enc_graph(train_rating_pairs, train_rating_values, train_sentiment_values, train_emotion_values, add_support=True)
        self.train_dec_graph = self._generate_dec_graph(train_rating_pairs)
        self.train_labels = self._make_labels(train_rating_values)
        self.train_truths = th.FloatTensor(train_rating_values).to(device)

        self.valid_s_graph, self.valid_e_graph = self.train_s_graph, self.train_e_graph
        self.valid_enc_graph = self.train_enc_graph 
        self.valid_dec_graph = self._generate_dec_graph(valid_rating_pairs)
        self.valid_labels = self._make_labels(valid_rating_values)
        self.valid_truths = th.FloatTensor(valid_rating_values).to(device)

#         self.test_enc_graph = self._generate_enc_graph(all_rating_pairs, all_train_rating_values, all_train_sentiment_values, all_train_emotion_values, add_support=True)
        self.test_enc_graph = self._generate_enc_graph(all_rating_pairs, all_train_rating_values, all_sentiment_values, all_emotion_values, add_support=True)
        self.test_dec_graph = self._generate_dec_graph(test_rating_pairs)
        self.test_labels = self._make_labels(test_rating_values)
        self.test_truths = th.FloatTensor(test_rating_values).to(device)
        
        


        
        print('......7: Graph 결과 출력')
        print("Train enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_enc_graph.number_of_nodes('user'), self.train_enc_graph.number_of_nodes('movie'),
            self._npairs(self.train_enc_graph)))
        print("Train dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_dec_graph.number_of_nodes('user'), self.train_dec_graph.number_of_nodes('movie'),
            self.train_dec_graph.number_of_edges()))
        print("Valid enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.valid_enc_graph.number_of_nodes('user'), self.valid_enc_graph.number_of_nodes('movie'),
            self._npairs(self.valid_enc_graph)))
        print("Valid dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.valid_dec_graph.number_of_nodes('user'), self.valid_dec_graph.number_of_nodes('movie'),
            self.valid_dec_graph.number_of_edges()))
        print("Test enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.test_enc_graph.number_of_nodes('user'), self.test_enc_graph.number_of_nodes('movie'),
            self._npairs(self.test_enc_graph)))
        print("Test dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.test_dec_graph.number_of_nodes('user'), self.test_dec_graph.number_of_nodes('movie'),
            self.test_dec_graph.number_of_edges()))

    
    
    def _make_labels(self, ratings):
        labels = th.LongTensor(np.searchsorted([i*0.5 for i in range(1,11)], ratings)).to(self._device)
        return labels
        
    def _npairs(self, graph):
        rst = 0
        for r in self.possible_rating_values:
            r = to_etype_name(r)
            rst += graph.number_of_edges(str(r))
        return rst
    
    def _generate_pair_value(self, rating_info):
        rating_pairs = (np.array([self.global_user_id_map[ele] for ele in rating_info["user_id"]],
                                 dtype=np.int64),
                        np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
                                 dtype=np.int64))
        rating_values = rating_info["rating"].values.astype(np.float32)
        sentiment_values = rating_info["sentiment"].values.astype(np.int16) 
        emotion_values = rating_info["emotion"].values.astype(np.int16)
        return rating_pairs, rating_values, sentiment_values, emotion_values

    def _generate_enc_graph(self, rating_pairs, rating_values, sentiment_values, emotion_values, add_support=False):

        data_dict = dict()
        num_nodes_dict = {'user': self._num_user, 'movie': self._num_movie}
        rating_row, rating_col = rating_pairs

        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('user', str(rating), 'movie'): (rrow, rcol),
                ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })

        if self.sentiment == True: 
            for rating in self.sentiment_rating_values:
                ridx = np.where(sentiment_values == rating)
                rrow = rating_row[ridx]
                rcol = rating_col[ridx]
                rating = to_etype_name(rating)
                data_dict.update({
                    ('user', str(rating), 'movie'): (rrow, rcol),
                    ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
                })

        if self.emotion == True:
            for rating in self.emotion_rating_values:
                ridx = np.where(emotion_values == rating)
                rrow = rating_row[ridx]
                rcol = rating_col[ridx]
                rating = to_etype_name(rating)
                data_dict.update({
                    ('user', str(rating), 'movie'): (rrow, rcol),
                    ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
                })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        

        # sanity check
        # assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)
            user_ci = []
            user_cj = []
            movie_ci = []
            movie_cj = []
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                user_ci.append(graph['rev-%s' % r].in_degrees())
                movie_ci.append(graph[r].in_degrees())
                if self._symm:
                    user_cj.append(graph[r].out_degrees())
                    movie_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    user_cj.append(th.zeros((self.num_user,)))
                    movie_cj.append(th.zeros((self.num_movie,)))
            user_ci = _calc_norm(sum(user_ci))
            movie_ci = _calc_norm(sum(movie_ci))
            if self._symm:
                user_cj = _calc_norm(sum(user_cj))
                movie_cj = _calc_norm(sum(movie_cj))
            else:
                user_cj = th.ones(self.num_user,)
                movie_cj = th.ones(self.num_movie,)
            graph.nodes['user'].data.update({'ci' : user_ci, 'cj' : user_cj})
            graph.nodes['movie'].data.update({'ci' : movie_ci, 'cj' : movie_cj})

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        user_movie_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_user, self.num_movie), dtype=np.float32)
        g = dgl.bipartite_from_scipy(user_movie_ratings_coo, utype='_U', etype='_E', vtype='_V')
        return dgl.heterograph({('user', 'rate', 'movie'): g.edges()}, 
                               num_nodes_dict={'user': self.num_user, 'movie': self.num_movie})

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie
    
    def _process_user_fea(self):
        top_critic = (self.user_info['top_critic'] == False).values.astype(np.float32)
        all_publisher_name = set(self.user_info['publisher_name'])
        publisher_map = {ele: i for i, ele in enumerate(all_publisher_name)}
        publisher_one_hot = np.zeros(shape=(self.user_info.shape[0], len(all_publisher_name)),
                                  dtype=np.float32)
        publisher_one_hot[np.arange(self.user_info.shape[0]),
                       np.array([publisher_map[ele] for ele in self.user_info['publisher_name']])] = 1
        user_features = np.concatenate([top_critic.reshape((self.user_info.shape[0], 1)),
                                        publisher_one_hot], axis=1)

        return user_features


    def _process_movie_fea(self):
        import torchtext
        TEXT = torchtext.legacy.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
        embedding = torchtext.vocab.GloVe(name='840B', dim=300)

        title_embedding = np.zeros(shape=(self.movie_info.shape[0], 300), dtype=np.float32)
        release_years = np.zeros(shape=(self.movie_info.shape[0], 1), dtype=np.float32)

        for idx, row in self.movie_info.iterrows():
            title_context = row['movie_title']
            year = row['year']

            # We use average of glove
            title_embedding[idx, :] = embedding.get_vecs_by_tokens(TEXT.tokenize(title_context)).numpy().mean(axis=0)
            release_years[idx] = float(year)   
        
        movie_features = np.concatenate((title_embedding,
                                 (release_years - 1950.0) / 100.0,
                                 self.movie_genre_dummies),axis=1) 
        return movie_features

    
    def _generate_sub_graph(self, rating_pairs, sentiment_values, emotion_values):

        num_nodes_dict = {'user': self._num_user, 'movie': self._num_movie}
        rating_row, rating_col = rating_pairs

        data_dict = dict()
        for rating in self.sentiment_rating_values:
            ridx = np.where(sentiment_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('user', str(rating), 'movie'): (rrow, rcol),
                ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })

        s_graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        data_dict = dict()
        for rating in self.emotion_rating_values:
            ridx = np.where(emotion_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('user', str(rating), 'movie'): (rrow, rcol),
                ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })

        e_graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        return s_graph, e_graph



if __name__ == '__main__':
    RottenMovie(train_data='./data/trainset.csv',
                test_data='./data/testset.csv',
                feature_data='./'
    )