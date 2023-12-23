import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Function for rule-based categorization (as a fallback)
def rule_based_categorization(query):
    # Example rules - these need to be tailored to your specific categories
    if "best" in query:
        return "Quality"
    elif "price" in query:
        return "Pricing"
    # Add more rules as needed
    return "General"

# Function to train and apply the classification model
def classify_queries(dataframe):
    # Assuming 'Content Type' column exists in your labeled dataset
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe['question'], dataframe['Content Type'], test_size=0.2, random_state=42)

    # Text preprocessing and classification pipeline
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Train the model
    model.fit(X_train, y_train)

    # Apply the model
    predicted_categories = model.predict(dataframe['question'])

    return predicted_categories

# Streamlit app
def main():
    st.title('Enhanced Query Categorization App')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if st.button('Categorize Queries'):
            # Check if the dataset is labeled for supervised learning
            if 'Content Type' in data.columns:
                result = classify_queries(data)
            else:
                # Fallback to rule-based categorization
                result = data['question'].apply(rule_based_categorization)

            data['Content Type'] = result
            st.write(data)

            # Export to CSV
            st.download_button(
                label="Download data as CSV",
                data=data.to_csv(index=False).encode('utf-8'),
                file_name='categorized_queries.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
