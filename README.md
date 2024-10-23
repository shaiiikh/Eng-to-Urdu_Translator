Eng-to-Urdu_Translator

Overview
This project implements an English-to-Urdu translation system using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models. The objective is to explore the capabilities and limitations of these models for language translation tasks, particularly in handling complex language pairs like English and Urdu.

Project Structure
.venv/: Contains Python libraries and the virtual environment setup for the project.
LSTM_model.keras: Trained LSTM model for the translation task.
RNN_model.keras: Trained RNN model for the translation task.
Translator.ipynb: Jupyter Notebook containing the code for training, evaluation, and translation using the RNN and LSTM models.
parallel-corpus.xlsx: Parallel corpus dataset used for training and testing, consisting of English and Urdu sentence pairs.

Objective
The project aims to:

Implement a many-to-many RNN model for English-to-Urdu translation.
Compare the performance of the RNN model with an LSTM-based model.
Address the limitations of RNNs in language translation using LSTM architectures.

Dataset
The translation model uses the parallel-corpus.xlsx file, which contains paired sentences in English and Urdu. This dataset is used to train, validate, and test the models.

Implementation Steps
Part 1: RNN Implementation
Data Preparation:
Cleaned the dataset by removing unnecessary columns, emojis, links, extra spaces, and duplicates.
Tokenized the sentences and created vocabularies for English and Urdu.
Padded sequences and split the data into training, validation, and test sets.
Model Architecture:
Built an RNN-based architecture using TensorFlow.
Trained the model with the cleaned dataset.
Evaluation:
Evaluated the RNN model using BLEU scores.
Provided example translations from the test set.

Part 2: Limitations of RNNs
Exploding/Vanishing Gradients: Struggles with long sequences due to gradient issues.
Capturing Long-Term Dependencies: Difficult to maintain context over many steps, especially in complex languages like Urdu.
Poor Performance on Large Datasets: Struggles with datasets that have diverse syntactic and grammatical variations.

Part 3: LSTM Implementation
LSTM Architecture:
Modified the RNN model by replacing RNN layers with LSTM layers.
Implemented input, forget, and output gates to retain relevant information.
Comparison and Evaluation:
Compared the performance of RNN and LSTM models using BLEU scores.
Showed that LSTM achieved higher scores, demonstrating better handling of long-term dependencies and complex datasets.
Results and Improvements
The LSTM model outperformed the RNN in terms of BLEU score, showing enhanced ability to handle long sequences and maintain context.
LSTM's gating mechanisms effectively addressed gradient issues and improved learning on large datasets.
Challenges and Future Work

Remaining Challenges:
Urdu's rich morphological structure poses challenges.
Contextual ambiguity and generalization issues remain.
Suggestions for Improvement:
Incorporate attention mechanisms to focus on relevant input segments.
Use transformer models to better handle long-range dependencies.
Consider data augmentation techniques and fine-tuning pre-trained models for further enhancement.

Requirements
To set up the environment, make sure to have the following installed:

Python 3.x
TensorFlow or PyTorch
Jupyter Notebook
Any other necessary libraries can be installed as you encounter ImportError messages (e.g., pip install <missing-library>).

Usage
Clone the repository.

Set up the virtual environment:
python -m venv .venv
source .venv/bin/activate (on Unix)
.venv\Scripts\activate (on Windows)
Install the required packages by running the necessary pip install commands as you run into ImportError messages.
Run Translator.ipynb to train, evaluate, and test the models.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
