# Project Notes

## Goal
Build a small recommendation system that demonstrates AI-based personalization for a service catalog while remaining easy to run on a Windows laptop.

## Design
The project uses neural collaborative filtering. Each customer and service is represented by a trainable embedding. The model computes the dot product of the embeddings, adds learned user and service biases, and applies a sigmoid output. Ratings are normalized from the 1-5 scale to 0-1 during training.

## Evaluation
Interactions are divided into 80% training and 20% test data with a fixed random seed. The program reports mean squared error (MSE) and mean absolute error (MAE). The normalized MAE is also multiplied by five to provide an approximate interpretation on the original rating scale.

## Cold start
An unknown customer has no learned embedding. The fallback therefore ranks services using:
`average_rating * log(1 + interaction_count)`.
This balances quality with a small popularity signal.

## Persistence and cache
The trained model is saved as `service_recommender.keras`. Future executions load the saved model. During a running session, an in-memory cache prevents repeated model loading.

## Development issues encountered
1. PowerShell initially blocked virtual-environment activation. The CurrentUser execution policy was changed to RemoteSigned.
2. TensorFlow version compatibility required Python 3.11 rather than Python 3.13 for the selected dependency range.
3. Keras model serialization initially failed because the custom model did not expose serializable configuration. `get_config()` and Keras registration were added.
4. `scikit-learn` was added after `train_test_split` produced a missing-module error.
5. The initial cache only prevented retraining within one process. Model loading from the saved `.keras` file was then added so later executions do not retrain unnecessarily.

## Ethical considerations
A real deployment should disclose personalization, obtain lawful/appropriate data, minimize unnecessary personal data, test for unequal recommendation quality, document limitations, provide human oversight where decisions have material consequences, and monitor model drift.
