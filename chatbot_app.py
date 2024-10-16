import streamlit as st
import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# Configure Azure AI Search
search_service_name = "your-search-service-name"
search_index_name = "your-search-index-name"
search_api_key = "your-search-api-key"

# Configure OpenAI
openai.api_key = "your-openai-api-key"

def search_azure_index(query):
    # Create a SearchClient
    credential = AzureKeyCredential(search_api_key)
    search_client = SearchClient(
        endpoint=f"https://{search_service_name}.search.windows.net/",
        index_name=search_index_name,
        credential=credential
    )

    # Perform the search
    results = search_client.search(query)
    return [result for result in results]

def generate_openai_response(prompt, search_results):
    # Prepare the prompt with search results
    full_prompt = f"{prompt}\n\nContext from search results:\n"
    for result in search_results:
        full_prompt += f"- {result['content']}\n"
    full_prompt += "\nBased on the above context, please provide an answer:"

    # Generate response using OpenAI
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=full_prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def main():
    st.title("AI-Powered Chatbot")

    # User input
    user_input = st.text_input("Ask a question:")

    if user_input:
        # Search Azure AI Search Index
        search_results = search_azure_index(user_input)

        # Generate response using OpenAI
        ai_response = generate_openai_response(user_input, search_results)

        # Display results
        st.subheader("AI Response:")
        st.write(ai_response)

        st.subheader("Search Results:")
        for result in search_results:
            st.write(f"- {result['content']}")

if __name__ == "__main__":
    main()