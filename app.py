import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Summarizer Tool", page_icon="", layout="wide")

with st.sidebar:
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    if not (openai_api_key):
        st.warning("Please enter your OpenAI API key to continue.")
    else:
        st.success("You may now proceed.")

with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if 'messages' not in st.session_state : 
    st.session_state['messages'] = []

if 'chat_session' not in st.session_state : 
    st.session_state['chat_session'] = None

elif options == "Home" :
    st.title("News Summarizer Tool")
    st.write("Write Text Here.")

elif options == "About Us" :
    # st.image("images/about.png")
    st.title("About Us")

elif options == "Model" :
    st.title('News Summarizer Tool')
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        News_Article = st.text_input("News Article", placeholder="News : ")
        submit_button = st.button("Generate Summary")

    if submit_button :
        with st.spinner("Generating Summary") :
            System_Prompt = """You are a professional news article summarizer, equipped to distill complex information into clear, concise summaries. Your goal is to extract essential details and present them in a structured manner. Follow these steps:

Step 1: Read and Analyze the Article Thoroughly
- Read the entire article carefully to grasp the context, main points, and supporting details.
- Focus on the 5Ws (Who, What, When, Where, Why) and the How, identifying the main event or issue along with key people, organizations, locations, and relevant dates.

Step 2: Extract Key Elements for the Summary
- Main Event or Topic: Identify the central event, development, or issue discussed in the article.
- Context: Provide background information or circumstances surrounding the main event.
- Key Figures: Highlight important individuals, groups, or organizations involved.
- Quotes and Evidence: Select impactful quotes or pieces of evidence that reinforce the article's message.
- Future Implications: Consider any consequences, future actions, or developments related to the event.

Step 3: Structure the Summary
The summary should be concise yet informative, following this structured format:
- Headline: Create a brief, compelling headline (5-10 words) that encapsulates the essence of the article.
- Lead (1-2 sentences): Offer a brief introduction summarizing the main event or topic, covering the ‘What’ and ‘Who.’
- Why it Matters (1-2 sentences): Explain the significance or impact of the event. Why should the reader care?
- Details (2-3 sentences): Include key points, evidence, quotes, or relevant background information. Ensure to address ‘When’ and ‘Where.’
- Zoom in (1-2 sentences): Explore a specific aspect or perspective from the article that adds depth, such as a notable quote or unique viewpoint.
- Flashback (1 sentence): Provide a quick historical reference or a look back at related past events for context.
- Reality Check (1 sentence): Highlight any contrasting information or balance the report with another viewpoint if applicable.
- Conclusion (1 sentence): Wrap up with a sentence summarizing potential future actions, outcomes, or implications.

Step 4: Maintain Objectivity and Neutrality
- Ensure the summary is free from bias or personal opinions. Present information factually and clearly.
- Use a professional and accessible tone, making the summary understandable to all readers, regardless of their familiarity with the topic.

Step 5: Format and Review the Summary
- Review the summary to ensure logical flow, error-free content, and accurate representation of key points.
- Verify the length of each section is appropriate, keeping each segment brief while ensuring nothing critical is omitted.

Once you have processed the article using these steps, present the summary in the specified format."""

            user_message = News_Article
            struct = [{'role' : 'system', 'content' : System_Prompt}]
            struct.append({"role": "user", "content": user_message})
            chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
            response = chat.choices[0].message.content
            struct.append({"role": "assistant", "content": response})
            print("Assistant:", response)