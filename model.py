import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
SERVICES_FILE = BASE_DIR / "data" / "services.csv"
INTERACTIONS_FILE = BASE_DIR / "data" / "interactions.csv"
MODEL_FILE = BASE_DIR / "service_recommender.keras"

_MODEL_CACHE = None


def load_data():
    services = pd.read_csv(SERVICES_FILE)
    interactions = pd.read_csv(INTERACTIONS_FILE)
    return services, interactions


@tf.keras.utils.register_keras_serializable(package="ServiceRecommender")
class RecommenderNet(tf.keras.Model):
    """Neural collaborative-filtering model with trainable embeddings."""

    def __init__(self, num_users, num_services, embedding_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.num_users = int(num_users)
        self.num_services = int(num_services)
        self.embedding_dim = int(embedding_dim)

        self.user_embedding = tf.keras.layers.Embedding(
            self.num_users, self.embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.user_bias = tf.keras.layers.Embedding(self.num_users, 1)
        self.service_embedding = tf.keras.layers.Embedding(
            self.num_services, self.embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.service_bias = tf.keras.layers.Embedding(self.num_services, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        service_vector = self.service_embedding(inputs[:, 1])
        service_bias = self.service_bias(inputs[:, 1])
        dot = tf.reduce_sum(user_vector * service_vector, axis=1, keepdims=True)
        score = dot + user_bias + service_bias
        return tf.nn.sigmoid(score)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.num_users,
            "num_services": self.num_services,
            "embedding_dim": self.embedding_dim,
        })
        return config


def prepare_training_data(interactions):
    users = sorted(interactions["user_id"].unique())
    services = sorted(interactions["service_id"].unique())
    user_to_index = {u: i for i, u in enumerate(users)}
    service_to_index = {s: i for i, s in enumerate(services)}

    x = np.array(
        [[user_to_index[u], service_to_index[s]]
         for u, s in zip(interactions["user_id"], interactions["service_id"])],
        dtype=np.int32,
    )
    y = interactions["rating"].astype("float32").to_numpy() / 5.0
    return x, y, user_to_index, service_to_index


def train_model(epochs=120):
    services, interactions = load_data()
    x, y, user_to_index, service_to_index = prepare_training_data(interactions)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    tf.random.set_seed(42)
    np.random.seed(42)

    model = RecommenderNet(
        len(user_to_index), len(service_to_index), embedding_dim=16
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    print("\nTraining model...")
    print(f"Training interactions: {len(x_train)}")
    print(f"Test interactions: {len(x_test)}")
    model.fit(x_train, y_train, batch_size=8, epochs=epochs, verbose=0)

    mse, mae = model.evaluate(x_test, y_test, verbose=0)
    print("\nModel evaluation")
    print("-" * 30)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f} (normalized scale)")
    print(f"Approximate MAE on 1-5 scale: {mae * 5:.2f} points")

    _ = model(tf.constant([[0, 0]], dtype=tf.int32))
    model.save(MODEL_FILE)
    print(f"\nModel saved to: {MODEL_FILE.name}")

    return model, services, interactions, user_to_index, service_to_index


def get_model():
    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    services, interactions = load_data()
    _, _, user_to_index, service_to_index = prepare_training_data(interactions)

    if MODEL_FILE.exists():
        print("\nLoading trained model...")
        model = tf.keras.models.load_model(
            MODEL_FILE,
            custom_objects={"RecommenderNet": RecommenderNet},
        )
        _MODEL_CACHE = (
            model, services, interactions, user_to_index, service_to_index
        )
    else:
        print("\nSaved model not found. Starting training...")
        _MODEL_CACHE = train_model()

    return _MODEL_CACHE


def recommend_for_user(user_id, top_n=5, category=None):
    model, services, interactions, user_to_index, service_to_index = get_model()

    if user_id not in user_to_index:
        merged = interactions.merge(services, on="service_id")
        scores = merged.groupby(
            ["service_id", "service_name", "category", "price"], as_index=False
        ).agg(
            avg_rating=("rating", "mean"),
            interactions=("rating", "count"),
        )
        scores["score"] = scores["avg_rating"] * np.log1p(scores["interactions"])
        if category:
            scores = scores[
                scores["category"].str.lower() == category.lower()
            ]
        return scores.sort_values("score", ascending=False).head(top_n), True

    rated = set(
        interactions.loc[
            interactions["user_id"] == user_id, "service_id"
        ]
    )
    candidates = services[~services["service_id"].isin(rated)].copy()
    candidates = candidates[candidates["service_id"].isin(service_to_index)]

    if category:
        candidates = candidates[
            candidates["category"].str.lower() == category.lower()
        ]

    if candidates.empty:
        return candidates, False

    user_index = user_to_index[user_id]
    pairs = np.array(
        [[user_index, service_to_index[s]] for s in candidates["service_id"]],
        dtype=np.int32,
    )
    predictions = model.predict(pairs, verbose=0).reshape(-1)
    candidates["predicted_score"] = predictions * 5.0
    return candidates.sort_values(
        "predicted_score", ascending=False
    ).head(top_n), False
