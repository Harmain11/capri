import re, os
from flask import Flask, render_template, request, jsonify
import openai
import pandas as pd
# For language detection
from langdetect import detect, DetectorFactory

# LangChain and FAISS imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Ensure deterministic language detection
DetectorFactory.seed = 0

# Inline API key (not recommended for production)
API_KEY = "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"
openai.api_key = API_KEY

app = Flask(__name__)

# Load and normalize CSV
df = pd.read_csv("data.csv")
df = df.rename(columns={"Categoría coloquial (PR)": "CategoriaPR"})
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()

# Build Documents for vector store
docs = []
for idx, row in df.iterrows():
    txt = (
        f"Pasillo: {row.Pasillo} | "
        f"Categoría: {row.Categoría} | "
        f"Coloquial: {row.CategoriaPR} | "
        f"Marca: {row.Marca} | "
        f"Área góndola: {row['Área en la góndola']} | "
        f"Ubicación frontal: {row['Ubicación según entrada principal']} | "
        f"Ubicación trasera: {row['Ubicación según entrada trasera']}"
    )
    docs.append(Document(page_content=txt, metadata={"row": idx}))

# Create embeddings and index
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
vector_store = FAISS.from_documents(docs, embeddings)

def retrieve_context(query: str, k: int = 5):
    """Retrieve the top-k most relevant documents for the query."""
    return vector_store.similarity_search(query, k=k)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    user_message = request.form.get("message", "").strip()
    if not user_message:
        return jsonify({"response": ""})

    # Detect language ('es' for Spanish, anything else defaults to English)
    try:
        lang = detect(user_message)
    except Exception:
        lang = 'en'

    # Retrieve context snippets
    results = retrieve_context(user_message, k=5)
    context_block = "\n\n".join([r.page_content for r in results])

    # Build system prompt based on detected language
    if lang == 'es':
        system_prompt = (
            "Eres Capri Assistant. Usa los siguientes datos del plano de la tienda para responder a la pregunta del usuario. "
            "Si el usuario pregunta en español, responde en español. Sé conciso y utiliza el mismo idioma.\n\n"
            f"{context_block}"
        )
    else:
        system_prompt = (
            "You are Capri Assistant. Use the following store-layout data to answer the user's question. "
            "If the user writes in English, respond in English. Be concise and use the same language the user used.\n\n"
            f"{context_block}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        # Use new OpenAI Python>=1.0 interface
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
        # Extract content from response
        bot_response = resp.choices[0].message.content.strip()
    except Exception as e:
        bot_response = f"Error contacting OpenAI: {e}"

    return jsonify({"response": bot_response})

if __name__ == "__main__":
    # For local development
    app.run(host="0.0.0.0", port=5000, debug=True)
