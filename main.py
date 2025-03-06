import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
from pathlib import Path
import tempfile
import base64
from audio_recorder_streamlit import audio_recorder
import networkx as nx
import matplotlib.pyplot as plt
from googletrans import Translator

# Load environment variables and configure API
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Dictionary of supported languages
LANGUAGES = {
    'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Italian': 'it',
    'Portuguese': 'pt', 'Russian': 'ru', 'Japanese': 'ja', 'Korean': 'ko', 'Chinese (Simplified)': 'zh-cn',
    'Hindi': 'hi', 'Tamil': 'ta', 'Telugu': 'te', 'Marathi': 'mr', 'Bengali': 'bn', 'Gujarati': 'gu',
    'Malayalam': 'ml', 'Kannada': 'kn', 'Punjabi': 'pa'
}

translator = Translator()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """
    Answer the question based on the context below. If you cannot find the answer in the context, say "I don't have enough information to answer that question."
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def text_to_speech(text, lang='en'):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(fp.name)
        return fp.name

def play_audio(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format='audio/mp3')

def translate_text(text, target_lang='en'):
    return translator.translate(text, dest=target_lang).text if target_lang != 'en' else text

def process_voice_input(lang='en'):
    audio_bytes = audio_recorder()
    if audio_bytes:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
            fp.write(audio_bytes)
            temp_path = fp.name
        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language=lang)
            os.unlink(temp_path)
            return text
    return None

def generate_mind_map(text):
    edges = []
    sentences = text.split('.')
    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) > 1:
            edges.append((words[0], words[-1]))
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10, font_weight='bold', node_size=2000)
    plt.title("Mind Map")
    st.pyplot(plt)

def main():
    st.set_page_config(page_title="Multilingual ChatPDF with Voice & Mind Map", layout="wide")
    st.title("📚 Multilingual Smart PDF Chat with Voice & Mind Map 🎙")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = "English"
    
    with st.sidebar:
        st.header("Settings")
        st.session_state.selected_language = st.selectbox("Select Language", options=list(LANGUAGES.keys()))
        st.header("Document Upload")
        pdf_docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                create_vector_store(text_chunks)
                st.session_state.processing_done = True
                st.success("✅ Documents processed successfully!")
    
    user_question = st.text_input("Ask your question:")
    voice_button = st.button("🎤 Voice Input")
    speak_response = st.checkbox("🔊 Voice Output", value=True)
    
    selected_lang_code = LANGUAGES[st.session_state.selected_language]
    if voice_button:
        with st.spinner("🎤 Listening..."):
            voice_text = process_voice_input(selected_lang_code)
            if voice_text:
                user_question = voice_text
                st.info(f"🎤 You said: {voice_text}")
    
    if user_question and st.session_state.processing_done:
        with st.spinner("Thinking..."):
            english_question = translate_text(user_question, 'en') if selected_lang_code != 'en' else user_question
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = vector_store.similarity_search(english_question)
            chain = get_conversation_chain()
            response = chain({"input_documents": docs, "question": english_question}, return_only_outputs=True)
            translated_response = translate_text(response["output_text"], selected_lang_code)
            st.session_state.chat_history.append({"question": user_question, "answer": translated_response})
            if speak_response:
                audio_path = text_to_speech(translated_response, selected_lang_code)
                play_audio(audio_path)
                Path(audio_path).unlink()
    
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"*Q:* {chat['question']}\n\n*A:* {chat['answer']}")
    
    st.subheader("Generate Mind Map")
    if st.button("Create Mind Map"):
        generate_mind_map(get_pdf_text(pdf_docs))

if __name__ == "__main__":
    main()