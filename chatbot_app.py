import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
import requests
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure AI Search settings
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
search_api_key = os.getenv("AZURE_SEARCH_API_KEY")

# Azure OpenAI settings
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

# OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_key)

# Function to generate embeddings
def generate_embedding(text):
    headers = {
        "Content-Type": "application/json",
        "api-key": openai_api_key
    }
    data = {
        "input": text
    }
    response = requests.post(
        f"{openai_endpoint}/openai/deployments/{openai_deployment}/embeddings?api-version=2023-05-15",
        headers=headers,
        json=data
    )
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    else:
        raise Exception(f"Error generating embedding: {response.text}")

# Function to perform hybrid search
def hybrid_search(query, business_id=None, top_k=5, score_threshold=0.02):
    search_client = SearchClient(endpoint=search_endpoint,
                                 index_name=index_name,
                                 credential=AzureKeyCredential(search_api_key))

    vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=top_k, fields="embedding")
    filter_expression = f"business_id eq '{business_id}'" if business_id else None

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select="review_id,business_id,review_content,location,date",
        top=top_k,
        filter=filter_expression
    )

    search_results = []
    for result in results:
        if result["@search.score"] >= score_threshold:
            search_results.append({
                "review_id": result["review_id"],
                "business_id": result["business_id"],
                "review_content": result["review_content"],
                "location": result["location"],
                "date": result["date"],
                "score": result["@search.score"]
            })

    return search_results

# System prompts
system_prompt = """
You are an intelligent search engine assistant integrated with Azure AI Search. Your role is to transform natural language user queries into optimized search phrases to extract the most relevant results from an index of customer reviews and feedback.

Follow these key guidelines when generating search terms:

Extract Root Terms: Focus on root words or short phrases that capture the essence of the query. Avoid unnecessary details unless they are critical to the search intent.

Relevance to Reviews: Assume the index contains reviews, ratings, and feedback, so there is no need to include terms like "feedback," "opinions," or "reviews" unless they are central to the specific query.

Precision Over Completeness: Generate terms that prioritize precision. Your goal is to return the most accurate search results while using the appropriate terms necessary to capture the full intent or context of the query. Unnecessary search terms will dilute results.

Context-Sensitive Simplification: Reduce the query to its core meaning, focusing on elements like product names, features, attributes, and services when relevant. Omit unnecessary details or extraneous modifiers that are unlikely to impact search relevance.

Variations for Ambiguous Queries: For queries with potential ambiguities, include common variations (e.g., "loud" when querying about "background music").

No Superfluous Content: Avoid adding terms that are already implicit in the query context (e.g., "customer," "guest," or "restaurant" unless distinguishing between different entities is necessary).

Format: Always return the search terms as a Python array in this format:
['term1', 'term2', ...]
"""

analysis_system_prompt = """
You're an incredibly insightful, optimistic, professional consultant for business owners evaluating customer feedback.
Be smart. You will be supplied a query from the owner, and reviews. Some review material will be extremely helpful and revelant, other parts won't be.
Make sure you directly answer the question asked by the owner, nothing more.
At the end of your analysis, provide the customer reviews you used to form your answer.
If none of the reviews are relevant, present the next closest relevant reviews and ask if the business owner would like to know about those instead.
"""

def extract_string(input_string):
    start_index = input_string.find('[')
    end_index = input_string.find(']')
    extracted_string = input_string[start_index + 1:end_index]
    return extracted_string

def main():
    st.title("Restaurant Review Analysis")

    # User input
    query = st.text_input("Enter your question about the restaurant:")
    business_id = st.text_input("Enter the business ID (optional):")

    if st.button("Search", type="primary"):
        if query:
            # Generate search terms
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            search_term = completion.choices[0].message.content
            extracted_string = extract_string(search_term)

            # Perform search
            results = hybrid_search(extracted_string, business_id=business_id, top_k=10)

            # # Display results
            # st.subheader("Search Results")
            # for i, result in enumerate(results, 1):
            #     st.write(f"Result {i}:")
            #     st.write(f"Review ID: {result['review_id']}")
            #     st.write(f"Business ID: {result['business_id']}")
            #     st.write(f"Location: {result['location']}")
            #     st.write(f"Date: {result['date']}")
            #     st.write(f"Overall Score: {result['score']}")
            #     st.write(f"Content: {result['review_content']}")
            #     st.write("---")

            # Prepare query and results for analysis
            query_and_results = f"The user asked: {query}\n"
            for i, result in enumerate(results, 1):
                query_and_results += f"\nCustomer Reviews {i}:\n"
                query_and_results += f"{result['review_content']}\n"

            # Generate analysis
            final_completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": analysis_system_prompt},
                    {"role": "user", "content": query_and_results}
                ],
                temperature=0.5
            )

            # Display analysis
            st.subheader("Analysis")
            st.write(final_completion.choices[0].message.content)

if __name__ == "__main__":
    main()