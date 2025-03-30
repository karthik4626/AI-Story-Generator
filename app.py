from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import re

# Function to extract word limit from prompt
def extract_word_limit(user_prompt):
    match = re.search(r'(\d+)\s*words?', user_prompt, re.IGNORECASE)  # Look for numbers followed by "words"
    return int(match.group(1)) if match else 500  # Default to 500 words

# Define Prompt Template
prompt = PromptTemplate.from_template(
    """
    You are an expert AI storyteller, skilled in crafting engaging, imaginative, and well-structured short stories.
    Your task is to generate a creative story based on the given prompt. Ensure the story has a clear beginning, middle, and end,
    with vivid descriptions, compelling characters, and an interesting plot twist.

    If a word limit is provided in the user's prompt, strictly follow it. If no limit is mentioned, generate a story of 500 words.

    Prompt: {user_prompt}
    Word Limit: {word_limit}

    Story:
    """
)

# Streamlit UI
st.title("AI Story Generator")
st.write("Enter a prompt and get an AI-generated short story!")

## Ollama tinyllama model
llm=OllamaLLM(model="tinyllama")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

user_prompt = st.text_area("Enter your story idea:")

if st.button("Generate Story") and user_prompt:
    word_limit = extract_word_limit(user_prompt)
    with st.spinner("Generating..."):
        story = chain.invoke({"user_prompt":user_prompt,"word_limit": word_limit})
    st.subheader("Generated Story:")
    st.write(story)

st.markdown("---")
st.write("Powered by TinyLLaMA, LangChain, and Streamlit")