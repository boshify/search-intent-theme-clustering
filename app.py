import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Function to perform clustering with BERT embeddings
def categorize_queries(dataframe):
    # Combine columns for clustering
    dataframe['combined'] = dataframe['type'] + " " + dataframe['modifier'] + " " + dataframe['question']
    
    # Load pre-trained BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Generating embeddings
    embeddings = model.encode(dataframe['combined'].tolist(), show_progress_bar=True)

    # KMeans clustering
    n_clusters = 5  # Adjust this based on your data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)

    # Assigning the cluster labels to the dataframe
    dataframe['Content Type'] = kmeans.labels_
    return dataframe

# Streamlit app
def main():
    st.title('Query Categorization App with BERT Embeddings')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if st.button('Categorize Queries'):
            result = categorize_queries(data)
            st.write(result)

            # Export to CSV
            st.download_button(
                label="Download data as CSV",
                data=result.to_csv(index=False).encode('utf-8'),
                file_name='categorized_queries.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
