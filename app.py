import os
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from PIL import Image
import time

@st.cache_data
def findanswer(Nand_url, Nand_question):
    if True:
      if Nand_url:
          index = None
          loader1 = PyPDFLoader(Nand_url)
          langchainembeddings = OpenAIEmbeddings(deployment="textembedding", chunk_size=1)

          index = VectorstoreIndexCreator(
                  # split the documents into chunks
                  text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
                  # select which embeddings we want to use
                  embedding=langchainembeddings,
                  # use Chroma as the vectorestore to index and search embeddings
                  vectorstore_cls=Chroma
              ).from_loaders([loader1])             
#          st.write("indexed PDF...AI finding answer....please wait")
      if Nand_question:
        answer = index.query(llm=llmgpt3, question=yourquestion, chain_type="map_reduce")
        return answer


          
image = Image.open('Wipro logo.png')
st.image(image,   width=100)

st.write("Learn best practices in Data Centre Sustainability")




os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_VERSION'] = '2023-03-15-preview'
os.environ['OPENAI_API_KEY'] = "052fc719df9e4771838f3295b2ef82a3"
os.environ['OPENAI_API_BASE'] = "https://openaistudio255.openai.azure.com/"


llmgpt3 = AzureOpenAI(      deployment_name="testdavanci", model_name="text-davinci-003" )
#llmchatgpt = AzureOpenAI(     deployment_name="esujnand", model_name="gpt-35-turbo" )

samplequestions = ["What is  Energy Star 4.0 Standard?", "What is RoHS Directive?", "What is Green IT?", "Benefits of greening IT?", "Holistic Approach to Green IT",
                   "Using IT: Environmentally Sound Practices", "Designing Green Computers", "Epeat" ]


with st.form("my_form"):

   myurl = st.text_input("What is the URL?", "https://sites.pitt.edu/~dtipper/2011/GreenPaper.pdf")

   yourquestion = st.selectbox(
    'Select',  samplequestions    )    

   # Every form must have a submit button.
   submitted = st.form_submit_button("Ask question")
   if submitted:
      #st.write("AI is looking for the answer...It will take atleast 2 mintutes... Answers will appear below....")
      Nandanswer = findanswer(myurl, yourquestion )
      st.write(Nandanswer)



