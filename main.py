import os
import pandas as pd
from PyPDF2 import PdfReader
import whisper
import docx
import ollama
import chromadb


# Function to process PDFs
def process_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to process TXT
def process_txt(file_path):
    with open(file_path, 'r') as file:
        return file.read()


# Function to process DOCX
def process_docx(file_path):
    doc = docx.Document(file_path)
    return " ".join([para.text for para in doc.paragraphs])


# Function to process MP3/MP4
def transcribe_audio(file_path, model="base"):
    audio_model = whisper.load_model(model)
    result = audio_model.transcribe(file_path)
    return result['text']


# Function to process XLS/XLSX
def process_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_string()


# Main Preprocessing Function
def preprocess(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == "pdf":
        return process_pdf(file_path)
    elif ext == "txt":
        return process_txt(file_path)
    elif ext in ["doc", "docx"]:
        return process_docx(file_path)
    elif ext in ["mp3", "mp4"]:
        return transcribe_audio(file_path)
    elif ext in ["xls", "xlsx"]:
        return process_excel(file_path)
    else:
        return "Unsupported file type"


client = chromadb.Client()
collection = client.create_collection(name="docs")

data_folder = "notes"

for file_name in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file_name)
    if not os.path.isfile(file_path):  # Check if the file exists
        print(f"File not found: {file_name}")
        continue

    try:
        # Preprocess file to extract content
        content = preprocess(file_path)
        if content != "Unsupported file type":
            # Get embeddings
            response = ollama.embeddings(model="mxbai-embed-large", prompt=content)
            embedding = response["embedding"]

            # Add to ChromaDB collection
            collection.add(
                ids=[file_name],  # Use file name as ID
                embeddings=[embedding],
                documents=[content]
            )
            print(f"Processed and stored embeddings for: {file_name}")
        else:
            print(f"Unsupported file type: {file_name}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

while True:
    prompt = input("> ")

    response = ollama.embeddings(
        prompt=prompt,
        model="mxbai-embed-large"
    )

    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=1
    )
    data = results['documents'][0][0]

    output = ollama.generate(
        model="llama3.1",
        prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
    )

    print(output['response'])