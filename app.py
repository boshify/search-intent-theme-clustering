import openai
import streamlit as st
import pandas as pd

# Function to categorize queries using OpenAI's GPT model
def categorize_queries_with_gpt(dataframe):
    categories = []
    for index, row in dataframe.iterrows():
        query = row['question']

        # Modify the prompt as needed to fit the categorization style
        prompt = f"Categorize the following query based on search intent themes: '{query}'"

        # Sending request to OpenAI API
        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-1106",  # or another suitable engine
                prompt=prompt,
                max_tokens=10  # Adjust as needed
            )
            category = response.choices[0].text.strip()
        except Exception as e:
            category = "Error: " + str(e)

        categories.append(category)

    dataframe['Content Type'] = categories
    return dataframe

# Streamlit app
def main():
    st.title('Query Categorization App with OpenAI GPT')

    # Load the OpenAI API key from Streamlit secrets
    openai.api_key = st.secrets["openai"]["api_key"]

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if st.button('Categorize Queries'):
            result = categorize_queries_with_gpt(data)
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
