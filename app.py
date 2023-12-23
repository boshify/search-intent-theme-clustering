import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Function to perform clustering
def categorize_queries(dataframe):
    # Combine columns for clustering
    dataframe['combined'] = dataframe['type'] + " " + dataframe['modifier'] + " " + dataframe['question']
    
    # Vectorizing the text data
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(dataframe['combined'])

    # Number of clusters - you may want to tune this
    n_clusters = 5
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)

    # Assigning the cluster labels to the dataframe
    dataframe['Content Type'] = model.labels_
    return dataframe

# Streamlit app
def main():
    st.title('Query Categorization App')

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
                data=result.to_csv().encode('utf-8'),
                file_name='categorized_queries.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
