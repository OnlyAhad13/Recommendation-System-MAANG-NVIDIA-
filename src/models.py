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
                 dropout_rate: float = 0.2, l2_reg: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.cross_layers = cross_layers
        self.deep_layers = deep_layers or [256, 128, 64]
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.cross_weights = []
        self.cross_biases = []
        self.deep_nets = [
            keras.layers.Dense(units, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg))
            for units in self.deep_layers
        ]
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        for i in range(self.cross_layers):
            self.cross_weights.append(self.add_weight(name=f'cross_w_{i}', shape=(input_dim, 1), initializer='glorot_uniform'))
            self.cross_biases.append(self.add_weight(name=f'cross_b_{i}', shape=(input_dim,), initializer='zeros'))
    
    def call(self, inputs, training=None):
        x0 = inputs
        xl = x0
        for i in range(self.cross_layers):
            xl_T = tf.expand_dims(xl, axis=2)
            w_xl = tf.tensordot(xl_T, self.cross_weights[i], axes=(1, 0))
            cross_term = tf.squeeze(w_xl, axis=2)
            xl = x0 * cross_term + self.cross_biases[i] + xl
        
        deep_out = inputs
        for layer in self.deep_nets:
            deep_out = layer(deep_out)
        
        return tf.concat([xl, deep_out], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({'cross_layers': self.cross_layers, 'deep_layers': self.deep_layers, 'dropout_rate': self.dropout_rate, 'l2_reg': self.l2_reg})
        return config


@keras.utils.register_keras_serializable()
class MultiTowerModel(keras.Model):
    """Two-tower architecture for user and item embeddings."""
    
    def __init__(self, config: ModelConfig, user_vocab: List[str], item_vocab: List[str], 
                 feature_specs: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.user_vocab = user_vocab
        self.item_vocab = item_vocab
        self.feature_specs = feature_specs
        
        self.user_lookup = keras.layers.StringLookup(vocabulary=user_vocab, mask_token=None)
        self.user_embedding = keras.layers.Embedding(len(user_vocab) + 1, config.embedding_dim)
        
        self.item_lookup = keras.layers.StringLookup(vocabulary=item_vocab, mask_token=None)
        self.item_embedding = keras.layers.Embedding(len(item_vocab) + 1, config.embedding_dim)
        
        self.user_tower = keras.Sequential([keras.layers.Dense(units, activation='relu') for units in config.user_tower_dims] + [keras.layers.Dense(config.embedding_dim)])
        self.item_tower = keras.Sequential([keras.layers.Dense(units, activation='relu') for units in config.item_tower_dims] + [keras.layers.Dense(config.embedding_dim)])
    
    def call(self, features, training=None):
        user_emb = None
        item_emb = None

        # Check if 'user_id' is provided before creating user embedding
        if 'user_id' in features:
            user_emb = self.user_tower(self.user_embedding(self.user_lookup(features['user_id'])))
        
        # Check if 'movie_id' is provided before creating item embedding
        if 'movie_id' in features:
            item_emb = self.item_tower(self.item_embedding(self.item_lookup(features['movie_id'])))
        
        return {'user_embedding': user_emb, 'item_embedding': item_emb}

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.to_dict(), "user_vocab": self.user_vocab, "item_vocab": self.item_vocab, "feature_specs": self.feature_specs})
        return config

    @classmethod
    def from_config(cls, config):
        model_config_dict = config.pop("config")
        config["config"] = ModelConfig(**model_config_dict)
        return cls(**config)


@keras.utils.register_keras_serializable()
class MultiTaskModel(tfrs.models.Model):
    """Multi-task model combining retrieval and ranking."""
    
    def __init__(self, config: ModelConfig, user_vocab: List[str], item_vocab: List[str],
                 feature_specs: Dict[str, Any], class_weights: Dict[int, float] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.encoder = MultiTowerModel(config, user_vocab, item_vocab, feature_specs)
        self.class_weights = class_weights

        self.retrieval_task = tfrs.tasks.Retrieval()
        
        self.dcn = DeepCrossNetwork(config.cross_layers, config.dnn_dims, config.dropout_rate, config.l2_reg)
        self.rating_head = keras.layers.Dense(1, name='rating_pred')
        self.ctr_head = keras.layers.Dense(1, activation='sigmoid', name='ctr_pred')
        
        self.rating_task = tfrs.tasks.Ranking(loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.RootMeanSquaredError()])
        self.ctr_task = tfrs.tasks.Ranking(loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy()])
    
    def call(self, data, training=None):
        features, _ = data if isinstance(data, tuple) else (data, None)
        emb = self.encoder(features, training=training)
        combined = tf.concat([emb['user_embedding'], emb['item_embedding']], axis=1)
        dcn_out = self.dcn(combined, training=training)
        
        return {**emb, 'rating_prediction': self.rating_head(dcn_out), 'ctr_prediction': self.ctr_head(dcn_out)}
    
    def compute_loss(self, data, training=False):
        features, labels = data if isinstance(data, tuple) else (data, data)
        pred = self(data, training=training)
        
        ret_loss = self.retrieval_task(pred['user_embedding'], pred['item_embedding'])
        rating_loss = self.rating_task(labels=labels['rating'], predictions=pred['rating_prediction'])
        
        ctr_loss = tf.constant(0.0)
        if 'y_implicit' in labels:
            sample_weight = None
            if self.class_weights:
                sample_weight = tf.where(tf.equal(labels['y_implicit'], 1.0), self.class_weights[1], self.class_weights[0])
            ctr_loss = self.ctr_task(labels=labels['y_implicit'], predictions=pred['ctr_prediction'], sample_weight=sample_weight)
        
        total_loss = (self.config.retrieval_weight * ret_loss + self.config.rating_weight * rating_loss + self.config.ctr_weight * ctr_loss)
        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config.to_dict(), "user_vocab": self.encoder.user_vocab, "item_vocab": self.encoder.item_vocab, "feature_specs": self.encoder.feature_specs, "class_weights": self.class_weights})
        return config

    @classmethod
    def from_config(cls, config):
        model_config_dict = config.pop("config")
        config["config"] = ModelConfig(**model_config_dict)
        return cls(**config)