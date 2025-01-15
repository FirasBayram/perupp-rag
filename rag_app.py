import streamlit as st
import pycountry
import faiss
import numpy as np
import pickle
import logging
import os
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)

def create_embedding(text, api_key):
    """Create embeddings using OpenAI API."""
    try:
        client = OpenAI()
        client.api_key = api_key
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"  # Using a more stable model
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Error creating embedding: {str(e)}")
        return None

def call_openai_api(prompt, context, api_key):
    """Send the prompt to OpenAI API using GPT-4 and return the response."""
    try:
        client = OpenAI(api_key=api_key)
        full_prompt = f"Context: {context}\n\nUser Query: {prompt}\n\nBased on the context and user query, provide a detailed travel plan:"
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return None

def load_index_and_texts(index_file="faiss_index_batch-3-LARGE.bin", text_file="texts.pkl"):
    """Load FAISS index and associated text data."""
    try:
        index = faiss.read_index(index_file)
        with open(text_file, "rb") as f:
            texts = pickle.load(f)
        return index, texts
    except Exception as e:
        st.error(f"Error loading index and texts: {str(e)}")
        return None, None

def search_index(index, query_embedding, k=5):
    """Search for the top k closest matches in the FAISS index."""
    try:
        distances, indices = index.search(query_embedding.reshape(1, -1), k)
        return distances, indices
    except Exception as e:
        st.error(f"Error searching index: {str(e)}")
        return None, None

# Get list of countries
countries = [country.name for country in pycountry.countries]

def main():
    st.title("Travel Planning App with RAG")

    # User Information
    st.header("User Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    country_residence = st.selectbox("Country of Residence", countries)
    city_residence = st.text_input("City of Residence")

    # Situational Data
    st.header("Situational Data")
    days_for_visit = st.number_input("# of Days for Visit", min_value=1, value=3, step=1)
    transportation = st.selectbox("Type of Transportation", ["Own Car", "Public Transport"])

    # Travel Constellation
    st.subheader("Traveling Constellation")
    num_people = st.number_input("# of People", min_value=1, value=1, step=1)
    num_children = st.number_input("# of Children under 10?", min_value=0, value=0, step=1)
    point_of_departure = st.text_input("Point of Departure")

    # Area of Interest
    st.header("Area of Interest")
    area_of_interest = st.multiselect(
        "Choose Areas of Interest",
        ["Literature", "Science and Engineering", "Design and Art", "Nature"],
        default=["Literature"]
    )

    # OpenAI API Key
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

    if openai_api_key and st.button("Generate Travel Plan"):
        if not city_residence or not point_of_departure:
            st.error("Please fill in all required fields.")
            return

        try:
            # Create the prompt
            normal_prompt = (
                f"Please suggest a travel plan based in Värmland based on the following details: "
                f"The user is {age} years old and identifies as {gender}. They reside in {city_residence}, "
                f"{country_residence} and plan to visit for {days_for_visit} days using {transportation}. "
                f"They will be traveling with {num_people} adults and {num_children} children under 10, "
                f"departing from {point_of_departure}. Their areas of interest are {', '.join(area_of_interest)}."
            )

            with st.spinner('Loading knowledge base...'):
                # Load FAISS index and texts
                index, texts = load_index_and_texts()
                if index is None or texts is None:
                    st.error("Failed to load knowledge base.")
                    return

                # Generate embedding for the query
                query_embedding = create_embedding(normal_prompt, openai_api_key)
                if query_embedding is None:
                    st.error("Failed to create embedding.")
                    return

                # Search for relevant context
                distances, indices = search_index(index, query_embedding)
                if distances is None or indices is None:
                    st.error("Failed to search knowledge base.")
                    return

                # Get relevant texts
                retrieved_texts = [texts[idx] for idx in indices[0] if 0 <= idx < len(texts)]
                combined_context = " ".join(retrieved_texts[:4])

            # Display prompt and context
            with st.expander("Show Generated Prompt and Context"):
                st.write("### Prompt")
                st.write(normal_prompt)
                st.write("### Retrieved Context")
                st.write(combined_context)

            # Generate travel plan
            with st.spinner('Generating your travel plan...'):
                response = call_openai_api(normal_prompt, combined_context, openai_api_key)
                if response:
                    st.write("### Your Travel Plan")
                    st.markdown(response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
