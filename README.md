# Text-Similarity-Technique-using-NLP-code

---

# Document Similarity Using Various Embedding Models

This repository contains code and models for document similarity analysis using different embeddings techniques, including Doc2Vec, Sentence-BERT, and Universal Sentence Encoder. It provides a comprehensive comparison of these methods for evaluating the semantic similarity between sentences and documents.

## Technologies Used

- **Gensim**: For training and using Doc2Vec models.
- **NLTK**: For tokenizing text.
- **Sentence-Transformers**: For using pre-trained Sentence-BERT models.
- **TensorFlow Hub**: For using the Universal Sentence Encoder.
- **Scipy**: For calculating cosine similarity.
- **Google Colab**: For executing the code in a cloud environment.

## Getting Started

### Installation

Before running the scripts, ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install gensim nltk sentence-transformers tensorflow tensorflow-hub scipy matplotlib
```

### Preparing the Data

In this project, we use a small set of sample documents for training and testing. The data is tokenized and processed to train the models and infer vectors for similarity analysis.

### Doc2Vec Model

#### Training

1. **Training Script**:
    ```python
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('punkt')
    
    # Sample data
    data = ["The movie is awesome. It was a good thriller",
            "We are learning NLP through GeeksforGeeks",
            "The baby learned to walk in the 5th month itself"]
    
    # Tokenizing the data
    tokenized_data = [word_tokenize(document.lower()) for document in data]
    
    # Creating TaggedDocument objects
    tagged_data = [TaggedDocument(words=words, tags=[str(idx)])
                   for idx, words in enumerate(tokenized_data)]
    
    # Training the Doc2Vec model
    model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=1000)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Save the trained model
    model.save("doc2vec_model.bin")
    ```
   
2. **Inference and Similarity Check**:
    ```python
    from gensim.models.doc2vec import Doc2Vec
    from nltk.tokenize import word_tokenize
    
    # Load the trained model
    loaded_model = Doc2Vec.load("doc2vec_model.bin")
    
    # Test with a new document
    new_document = "The baby was laughing and playing"
    inferred_vector = loaded_model.infer_vector(word_tokenize(new_document.lower()))
    
    # Find most similar documents
    similar_documents = loaded_model.dv.most_similar([inferred_vector], topn=len(loaded_model.dv))
    
    # Print the most similar documents
    for index, score in similar_documents:
        print(f"Document {index}: Similarity Score: {score}")
    ```

### Sentence-BERT Model

#### Usage

1. **Embedding and Similarity Calculation**:
    ```python
    from sentence_transformers import SentenceTransformer
    from scipy.spatial import distance
    
    # Load Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Sample sentences
    sentences = ["The movie is awesome. It was a good thriller",
                 "We are learning NLP through GeeksforGeeks",
                 "The baby learned to walk in the 5th month itself"]
    
    test = "I liked the movie."
    test_vec = model.encode([test])[0]
    
    for sent in sentences:
        similarity_score = 1 - distance.cosine(test_vec, model.encode([sent])[0])
        print(f'\nFor "{sent}"\nSimilarity Score = {similarity_score}')
    ```

2. **Saving and Loading the Model**:
    ```python
    import pickle
    
    def save_model(model, filename):
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
    
    def load_model(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    
    # Save the model
    save_model(model, 'SBERT.pkl')
    
    # Load the model
    loaded_model = load_model('SBERT.pkl')
    ```

### Universal Sentence Encoder

#### Usage

1. **Embedding and Similarity Calculation**:
    ```python
    import tensorflow as tf
    import tensorflow_hub as hub
    from scipy.spatial import distance
    
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    
    def embed(input):
        return model(input)
    
    test = ["I liked the movie very much"]
    test_vec = embed(test)
    
    sentences = [["The movie is awesome and It was a good thriller"],
                 ["We are learning NLP through GeeksforGeeks"],
                 ["The baby learned to walk in the 5th month itself"]]
    
    for sent in sentences:
        similarity_score = 1 - distance.cosine(test_vec[0, :], embed(sent)[0, :])
        print(f'\nFor {sent}\nSimilarity Score = {similarity_score}')
    ```

2. **Saving and Loading Embeddings**:
    ```python
    import numpy as np
    
    # Compute embeddings
    test_vec = embed(["I liked the movie very much"])
    sentences_vecs = [embed([sent])[0] for sent in sentences]
    
    # Save embeddings
    np.savez('USE_embeddings.npz', test=test_vec, sentences=sentences_vecs)
    
    # Load embeddings
    data = np.load('USE_embeddings.npz')
    test_vec = data['test']
    sentences_vecs = data['sentences']
    ```

3. **Testing USE**:
    ```python
    # Define a function to compute cosine similarity
    def cosine_similarity(vec1, vec2):
        return 1 - distance.cosine(vec1, vec2)
    
    # New test sentence
    new_test = ["I enjoyed the film a lot"]
    
    sentences = [
        ["The movie is nice and It was a good thriller"],
        ["We are learning NLP through GeeksforGeeks"],
        ["The baby learned to walk in the 5th month itself"]
    ]
    
    # Compute the new test sentence embedding
    new_test_vec = embed(new_test)[0]

    # Compute similarity scores for the new test sentence
    for i, sent_vec in enumerate(sentences_vecs):
        similarity_score = cosine_similarity(new_test_vec, sent_vec)
        print(f'\nFor sentence {i+1}: {sentences[i]}')
        print(f'Similarity Score = {similarity_score}')
    ```
## Full Code with Explanation

1. See the full code "Text_Similarity_Techniques.ipynb" for detailed explanation
   
## Running the Code

1. Clone the repository:
    ```bash
    git clone https://github.com/shukdevtroy/Text-Similarity-Technique-using-NLP-code.git
    cd Text-Similarity-Technique-using-NLP-code
    ```

2. Run the Text_Similarity_Techniques.ipynb in Google Colab 

## Acknowledgements

- Gensim for Doc2Vec implementation
- Sentence-Transformers for pre-trained models
- TensorFlow Hub for Universal Sentence Encoder

---

Feel free to modify the file paths and other details according to your specific setup and requirements.
