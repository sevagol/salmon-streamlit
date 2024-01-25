import streamlit as st
from openai import OpenAI
import openai
from pinecone import Pinecone
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from collections import Counter

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def create_query_body(context, tokenizer, model):
    # Tokenize the context
    inputs = tokenizer(
        context, padding=True, truncation=True,
        max_length=512,
    )

    # Encode the context to obtain dense vector
    emb = model.encode(context)

    # Calculate the sparse vector using Counter
    sparse_vec = dict(Counter(inputs['input_ids']))

    # Ensure that the values in the sparse vector are of float type
    sparse_vec = {
        'indices': list(sparse_vec.keys()),
        'values': [float(value) for value in sparse_vec.values()]
    }

    # Build the query body
    query_body = {
        'namespace': 'example-namespace',
        'top_k': 3,
        'vector': emb.tolist(),
        # 'sparse_vector': sparse_vec,
    }

    return query_body


def query_pinecone_index(query):
    pc = Pinecone(st.secrets.pinecone_key)
    index = pc.Index("salmon-index")
    query_body = create_query_body(query, tokenizer, model)
    response = index.query(**query_body, include_metadata=True)
    
    # Extract articles, headers, and dates from the matches
    articles = [match['metadata']['article'] for match in response['matches']]
    headers = [match['metadata']['header'] for match in response['matches']]
    dates = [match['metadata']['date'] for match in response['matches']]
    
    # Combine articles, headers, and dates into a structured result
    structured_result = []
    for date, header, article in zip(dates, headers, articles):
        result_item = {
            'date_of_publish': date,
            'name of the article': header,
            'article': article
        }
        structured_result.append(result_item)
    
    return structured_result

def create_prompt(question, document_content):
    return 'You are given a document and a question. Your task is to answer the question based on the document.\n\n' \
           'Document:\n\n' \
           f'{document_content}\n\n' \
           f'Question: {question}'

def get_answer_from_openai(question):
    client = OpenAI(
    # This is the default and can be omitted
    api_key=st.secrets.openai_key)
    document_content = query_pinecone_index(question)
    prompt = create_prompt(question, document_content)
    print(f'Prompt:\n\n{prompt}\n\n')
    completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="gpt-3.5-turbo",
)
    return completion.choices[0].message.content

st.set_page_config(page_title="Chat with the Streamlit about salmon industry", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Trends in metals mining 💬")
st.info("Here you can ask chatbot about recent trends in salmon industry", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question here"}
    ]


# if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
#         st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_answer_from_openai(prompt)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) # Add response to message history