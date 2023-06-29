# transfer-learning
## Pollen Classification Improvement through Weakly Supervised Transfer Learning

This repository presents a novel approach to enhance the classification of pollen using a weakly supervised transfer learning technique. By leveraging transfer learning, our method allows for significant improvement in pollen classification accuracy with limited labeled data.

The src folder contains the following key components:

1. "__crossvalidate_CNN.py__": This script facilitates the evaluation of a Convolutional Neural Network (CNN) using nested cross-validation. It enables robust testing and assessment of CNN's performance.
2. "__train_model.py__": With this script, you can train the final model using all available data and carefully selected hyperparameters determined in step 1. It ensures optimal training of the model for subsequent classification tasks.
3. "__retrain_model.py__": Here, you'll find a script that retrains the model obtained in step 2 by incorporating additional data, usually employed as ground truth. This step is crucial for refining and improving the model's performance further. Detailed information can be found in the folder files, paper titled: "ola2023".
4. "__calculate_correlation_distributions.py__": This script calculates the correlation distribution, taking into account the uncertainty introduced by the models and the ground truth. This analysis provides insights into the reliability and consistency of the classification results. More information about the uncertainty and correlation distribution can be found in the folder files, paper titled: "stoten2022".

Additionally, all these scripts rely on complementary scripts located in the Libraries folder, which provide necessary functions and utilities.

By employing this repository, researchers and practitioners in the field of pollen classification can leverage weakly supervised transfer learning to enhance their classification models and achieve more accurate results no matter which device they use.
