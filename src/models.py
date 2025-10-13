"""
Model architectures for the recommendation system.
"""

from typing import Dict, Any, List
import tensorflow as tf
from tensorflow import keras
import tensorflow_recommenders as tfrs

from .config import ModelConfig


@keras.utils.register_keras_serializable()
class DeepCrossNetwork(keras.Model):
    """Deep & Cross Network for feature interaction learning."""
    
    def __init__(self, cross_layers: int = 3, deep_layers: List[int] = None, 
                 dropout_rate: float = 0.2, l2_reg: float = 1e-5):
        super().__init__()
        self.cross_layers = cross_layers
        self.deep_layers = deep_layers or [256, 128, 64]
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.cross_weights = []
        self.cross_biases = []
        self.deep_nets = []
        
        for units in self.deep_layers:
            self.deep_nets.extend([
                keras.layers.Dense(units, activation='relu', 
                                 kernel_regularizer=keras.regularizers.l2(l2_reg)),
                keras.layers.Dropout(dropout_rate)
            ])
    
    def build(self, input_shape):
        super().build(input_shape)
        input_dim = input_shape[-1]
        
        for i in range(self.cross_layers):
            self.cross_weights.append(self.add_weight(
                name=f'cross_w_{i}', shape=(input_dim, 1),
                initializer='glorot_uniform', trainable=True
            ))
            self.cross_biases.append(self.add_weight(
                name=f'cross_b_{i}', shape=(input_dim,),
                initializer='zeros', trainable=True
            ))
    
    def call(self, inputs, training=None):
        x0, xl = inputs, inputs
        
        # Cross layers
        for i in range(self.cross_layers):
            xl = x0 * tf.matmul(xl, self.cross_weights[i]) + self.cross_biases[i] + xl
        
        # Deep layers
        deep_out = inputs
        for layer in self.deep_nets:
            deep_out = layer(deep_out, training=training)
        
        return tf.concat([xl, deep_out], axis=1)


@keras.utils.register_keras_serializable()
class MultiTowerModel(keras.Model):
    """Two-tower architecture for user and item embeddings."""
    
    def __init__(self, config: ModelConfig, user_vocab: List[str], item_vocab: List[str], 
                 feature_specs: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # User and item embeddings
        self.user_lookup = keras.layers.StringLookup(vocabulary=user_vocab, mask_token=None)
        self.user_embedding = keras.layers.Embedding(len(user_vocab) + 1, config.embedding_dim)
        
        self.item_lookup = keras.layers.StringLookup(vocabulary=item_vocab, mask_token=None)
        self.item_embedding = keras.layers.Embedding(len(item_vocab) + 1, config.embedding_dim)
        
        # Feature layers
        self.feature_layers = {}
        for name, spec in feature_specs.items():
            if spec['type'] == 'categorical':
                self.feature_layers[name] = keras.layers.Embedding(
                    min(spec['vocab_size'], 10000) + 1, config.embedding_dim // 2
                )
            else:
                self.feature_layers[name] = keras.layers.Dense(
                    config.embedding_dim // 2, activation='relu'
                )
        
        # User tower
        self.user_tower = keras.Sequential([
            *[layer for units in config.user_tower_dims 
              for layer in [
                  keras.layers.Dense(units, activation='relu'),
                  keras.layers.BatchNormalization(),
                  keras.layers.Dropout(config.dropout_rate)
              ]],
            keras.layers.Dense(config.embedding_dim)
        ])
        
        # Item tower
        self.item_tower = keras.Sequential([
            *[layer for units in config.item_tower_dims 
              for layer in [
                  keras.layers.Dense(units, activation='relu'),
                  keras.layers.BatchNormalization(),
                  keras.layers.Dropout(config.dropout_rate)
              ]],
            keras.layers.Dense(config.embedding_dim)
        ])
    
    def call(self, features, training=None):
        user_emb = None
        item_emb = None
        
        if 'user_id' in features:
            user_emb = self.user_embedding(self.user_lookup(features['user_id']))
            if len(user_emb.shape) == 3:
                user_emb = tf.squeeze(user_emb, axis=1)
            user_emb = self.user_tower(user_emb, training=training)
        
        if 'movie_id' in features:
            item_emb = self.item_embedding(self.item_lookup(features['movie_id']))
            if len(item_emb.shape) == 3:
                item_emb = tf.squeeze(item_emb, axis=1)
            item_emb = self.item_tower(item_emb, training=training)
        
        return {'user_embedding': user_emb, 'item_embedding': item_emb}


@keras.utils.register_keras_serializable()
class MultiTaskModel(tfrs.models.Model):
    """Multi-task model combining retrieval and ranking."""
    
    def __init__(self, config: ModelConfig, user_vocab: List[str], item_vocab: List[str],
                 feature_specs: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.encoder = MultiTowerModel(config, user_vocab, item_vocab, feature_specs)
        
        # Item vocabulary for retrieval
        self.item_vocab_list = item_vocab
        
        # Retrieval task
        self.retrieval_task = tfrs.tasks.Retrieval()
        
        # Ranking components
        self.dcn = DeepCrossNetwork(config.cross_layers, config.dnn_dims, 
                                   config.dropout_rate, config.l2_reg)
        self.rating_head = keras.layers.Dense(1, name='rating_pred')
        self.ctr_head = keras.layers.Dense(1, activation='sigmoid', name='ctr_pred')
        
        # Ranking tasks
        self.rating_task = tfrs.tasks.Ranking(
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.RootMeanSquaredError()]
        )
        self.ctr_task = tfrs.tasks.Ranking(
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy()]
        )
    
    def call(self, features, training=None):
        emb = self.encoder(features, training=training)
        combined = tf.concat([emb['user_embedding'], emb['item_embedding']], axis=1)
        dcn_out = self.dcn(combined, training=training)
        
        return {
            **emb,
            'rating_prediction': self.rating_head(dcn_out),
            'ctr_prediction': self.ctr_head(dcn_out)
        }
    
    def compute_loss(self, features, training=False):
        pred = self(features, training=training)
        
        # Retrieval loss
        ret_loss = self.retrieval_task(pred['user_embedding'], pred['item_embedding'])
        
        # Rating loss
        rating_loss = tf.constant(0.0)
        if 'rating' in features:
            labels = features['rating']
            if len(labels.shape) == 1:
                labels = tf.expand_dims(labels, -1)
            rating_loss = self.rating_task(labels, pred['rating_prediction'])
        
        # CTR loss
        ctr_loss = tf.constant(0.0)
        if 'y_implicit' in features:
            labels = features['y_implicit']
            if len(labels.shape) == 1:
                labels = tf.expand_dims(labels, -1)
            ctr_loss = self.ctr_task(labels, pred['ctr_prediction'])
        
        # Weighted combination - retrieval is prioritized!
        total_loss = (self.config.retrieval_weight * ret_loss + 
                     self.config.rating_weight * rating_loss + 
                     self.config.ctr_weight * ctr_loss)
        
        return total_loss

