import streamlit as st
import pycountry
import faiss
import numpy as np
import pickle
import logging
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up OpenAI API client (will be passed later)
def call_openai_api(prompt, context, api_key):
    """Send the prompt to OpenAI API using the fine-tuned model and return the response."""
    client = OpenAI(api_key=api_key)
    try:
        # Combine the context with the prompt before sending it to OpenAI
        full_prompt = f"Context: {context}\n\n{prompt}"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# FAISS Retrieval Functions
def load_index_and_texts(index_file="faiss_index_batch-3-LARGE.bin", text_file="texts.pkl"):
    """Load FAISS index and associated text data."""
    index = faiss.read_index(index_file)
    with open(text_file, "rb") as f:
        texts = pickle.load(f)
    return index, texts

def search_index(index, query_embedding, k=5):
    """Search for the top k closest matches in the FAISS index."""
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    return distances, indices

# Get list of countries (use pycountry library)
countries = [country.name for country in pycountry.countries]

# Streamlit app content
def main():
    st.title("Travel Planning App with RAG")

    # Step 1: Collect user information and travel preferences
    st.header("User Information")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    country_residence = st.selectbox("Country of Residence", countries)
    city_residence = st.text_input("City of Residence")

    st.header("Situational Data")
    days_for_visit = st.number_input("# of Days for Visit", min_value=1, step=1)
    transportation = st.selectbox("Type of Transportation", ["Own Car", "Public Transport"])

    st.subheader("Traveling Constellation")
    num_people = st.number_input("# of People", min_value=1, step=1)
    num_children = st.number_input("# of Children under 10?", min_value=0, step=1)
    point_of_departure = st.text_input("Point of Departure")

    st.header("Area of Interest")
    area_of_interest = st.multiselect(
        "Choose Areas of Interest",
        ["Literature", "Science and Engineering", "Design and Art", "Nature"],
        default=["Literature"]  # Set default interests if necessary
    )

    # Step 2: Enter OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

    # Step 3: Generate travel plan only after "Generate Travel Plan" button is pressed
    if openai_api_key:
        if st.button("Generate Travel Plan"):
            normal_prompt = (
                f"Please suggest a travel plan based in Värmland based on the following details: "
                f"The user is {age} years old and identifies as {gender}. They reside in {city_residence}, "
                f"{country_residence} and plan to visit for {days_for_visit} days using {transportation}. "
                f"They will be traveling with {num_people} adults and {num_children} children under 10, "
                f"departing from {point_of_departure}. Their areas of interest are {', '.join(area_of_interest)}."
            )

            # Load FAISS index and texts
            index, texts = load_index_and_texts()

            # Generate the embedding for the user's query (which is the travel plan prompt)
            response = OpenAI(api_key=openai_api_key).embeddings.create(input=[normal_prompt], model="text-embedding-ada-002")
            query_embedding = np.array(response.data[0].embedding)

            # Search the FAISS index for relevant documents
            distances, indices = search_index(index, query_embedding)

            # Get the most relevant texts based on the search
            retrieved_texts = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if 0 <= idx < len(texts):
                    retrieved_texts.append(texts[idx])

            # Combine the retrieved texts as context
            combined_context = " ".join(retrieved_texts[:4])  # Limit context to top 4 relevant texts

            # Display the generated prompt and context
            st.write("### Generated Prompt and Context for LLM")
            st.write(f"Prompt: {normal_prompt}")
            st.write(f"Context: {combined_context}")

            # Display spinner while waiting for the response
            with st.spinner('Generating your travel plan...'):
                # Send the normal prompt along with the context to OpenAI's API
                response = call_openai_api(normal_prompt, combined_context, openai_api_key)

            # Display the result after the spinner ends
            st.write("### Travel Plan Response from LLM")
            st.write(response)

# Run the app
if __name__ == '__main__':
    main()
