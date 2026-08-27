from fastapi import FastAPI, UploadFile, File, HTTPException
import os
from pydantic import BaseModel

from gemini_service import GeminiService
from analysis_engine import AnalysisEngine
from ml_engine import MachineLearningEngine
from rag_engine import RAGEngine
from embedding_service import EmbeddingService
from document_processor import DocumentProcessor

import pandas as pd

app = FastAPI(
    title="Enterprise AI Analytics Platform",
    description="AI-powered data analysis, machine learning, and RAG API",
    version="1.0"
)

# Load dataset
# df = pd.read_csv("datasets/employees.csv")
# df.columns = df.columns.str.lower()

# Initialize services
gemini = GeminiService()
# analysis_engine = AnalysisEngine(df)
# ml_engine = MachineLearningEngine(df)
df = None
analysis_engine = None
ml_engine = None
dataset_profile = None
rag_engine = RAGEngine()
gemini = GeminiService()
processor = DocumentProcessor()
embedding_service = EmbeddingService()

class QuestionRequest(BaseModel):
    question: str

# Sending information to the model
def create_dataset_profile(df):
    profile = {
        "numeric_columns": [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ],
        "categorical_columns": [
            col for col in df.columns
            if not pd.api.types.is_numeric_dtype(df[col])
        ],
        "rows": len(df),
        "columns": {}
    }

    for column in df.columns:

        column_info = {
            "dtype": str(df[column].dtype),
            "missing_values": int(df[column].isna().sum()),
            "unique_values": int(df[column].nunique()),
            "sample_values": (
                df[column]
                .dropna()
                .head(5)
                .tolist()
            )
        }

        # Datetime columns
        if pd.api.types.is_datetime64_any_dtype(df[column]):

            column_info["statistics"] = {
                "earliest": str(df[column].min()),
                "latest": str(df[column].max())
            }

        # Categorical/text columns
        else:
            column_info["top_values"] = (
                df[column]
                .value_counts()
                .head(5)
                .to_dict()
            )

        profile["columns"][column] = column_info

    return profile

def explain_ML_result(question, result):

    explanation = gemini.generate(
        f"""
        You are an AI machine learning assistant. 
        Your job is to explain machine learning results to the user.
        Answer the user's question using the result below.

        Rules:
        - Summarize the overall purpose of the model in one sentence.
        - Highlight the most important evaluation metrics only (accuracy, precision, recall, F1, MAE, RMSE, R², silhouette score depending on the task).
        - Do not explain what every metric means unless the user asks.
        - Do not describe internal implementation details unless relevant.
        - If feature importance is provided, list only the top 3-5 most important features.
        - Explain feature importance as relationships or patterns, not as guaranteed causes.
        - If the model performance is poor, mention that results may be unreliable.
        - Keep the response concise and focused on useful insights.
        - Use bullet points for metrics and findings.
        - Do not explain every field in the JSON.
        - Do not define basic machine learning terms unless asked.
        - Summarize the important findings.
        - Focus on actionable insights.
        - Do not mention file paths, model objects, preprocessing steps, or technical metadata unless requested.
        - If the result is an error give a short one sentence response simply stating what the error is.

        Question:
        {question}

        Result:
        {result}
        """
    )

    return explanation.text

def explain_result(question, result):

    explanation = gemini.generate(
        f"""
        Answer the user's question using the result below.

        Rules:
        - Answer the user's question directly.
        - Explain the meaning of the result, not the code used to calculate it.
        - Highlight important patterns, trends, or comparisons.
        - Do not repeat raw JSON unless necessary.
        - Keep numerical explanations concise.
        - Use tables or bullet points when comparing multiple values.
        - If the result represents an aggregation (average, sum, count, maximum, etc.), explain what it means in the context of the dataset.
        - Avoid unnecessary definitions of basic statistical concepts.
        - If the result is unclear or insufficient, state what additional analysis may be needed.
        - If the result is an error give a short one sentence response simply stating what the error is.
        - Be concise, avoid giving a long answer.

        Question:
        {question}

        Result:
        {result}
        """
    )

    return explanation.text

conversation = []
conversation_summary = ""

@app.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    global df
    global analysis_engine
    global ml_engine
    global dataset_profile

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        file.filename
    )

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Load uploaded CSV
    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = df.columns.str.lower()

    # Create engines using the uploaded dataset
    analysis_engine = AnalysisEngine(df)
    ml_engine = MachineLearningEngine(df)

    # Create dataset profile
    dataset_profile = create_dataset_profile(df)

    return {
        "message": "CSV uploaded successfully.",
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns)
    }

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):

    # Make sure the uploaded file is actually a PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        file.filename
    )

    # Save uploaded PDF
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Process PDF
    documents = processor.load_pdf(file_path)

    # Split into chunks
    chunks = processor.chunk_documents(documents)

    # Create embeddings/vector database
    vector_store = embedding_service.create_vector_store(chunks)

    # Save vector database
    embedding_service.save_vector_store(
        vector_store,
        path="vector_db"
    )

    return {
        "message": "PDF uploaded and processed successfully.",
        "filename": file.filename,
        "chunks": len(chunks)
    }

@app.get("/")
def root():
    return {
        "message": "Enterprise AI Analytics Platform API is running"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if df is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV dataset before asking questions."
        )

    question = request.question

    if len(conversation) > 8:
        conversation[:] = conversation[-4:]
    
    conversation.append(
    {
        "role": "user",
        "content": question
    })

    # Let Gemini determine what the user wants
    analysis_request = gemini.create_analysis_request(
        question,
        dataset_profile,
        conversation
    )

    print("Gemini returned:")
    print(analysis_request)
    print()

    response_data = {}

    # Analysis
    if analysis_request.intent == "analysis":

        result = analysis_engine.execute(analysis_request)

        response_data = {
            "intent": "analysis",
            "result": result
        }

    # Machine learning
    elif analysis_request.intent == "machine_learning":

        if analysis_request.task in ["classification", "regression"]:
            if not analysis_request.target_column:
                response_data = {
                    "intent": "machine_learning",
                    "error": (
                        "Classification and regression require "
                        "a target column."
                    )
                }
            
            if analysis_request.target_column not in df.columns:
                response_data = {
                    "intent": "machine_learning",
                    "error": (
                        f"Target column '{analysis_request.target_column}' "
                        f"does not exist. "
                        f"Available columns: {list(df.columns)}"
                    )
                }

            result = ml_engine.execute(analysis_request)

            response_data = {
                "intent": "machine_learning",
                "task": analysis_request.task,
                "result": result
            }

        elif analysis_request.task == "clustering":
            
            result = ml_engine.execute(analysis_request)

            response_data = {
                "intent": "machine_learning",
                "task": "clustering",
                "result": result
            }

        else:
            response_data = {
                "intent": "machine_learning",
                "error": (
                    "I don't know how to perform that "
                    "machine learning task yet."
                )
            }

    # RAG
    elif analysis_request.intent == "rag":

        context = rag_engine.retrieve_context(question)

        explanation = gemini.generate(
            f"""
            You are a helpful AI assistant.

            Answer ONLY using the context below.
            Retrieve information relevant to all concepts mentioned in the question.
            If the question asks for a comparison, find information about each concept and explain the differences.
            Combine information from multiple sections when necessary.
            If the answer is not contained in the context, say you do not know.

            Context:
            {context}

            Question:
            {question}
            """
        )

        response_data = {
            "intent": "rag",
            "answer": explanation.text
        }

    # Explanation
    elif analysis_request.intent == "explanation":

        conversation_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in conversation
        )

        explanation = gemini.generate(
            f"""
            You are a data analyst assistant.

            Use the dataset information and conversation history below to answer
            the user's question.

            Dataset:
            {dataset_profile}

            Conversation:
            {conversation_text}

            Question:
            {question}
            """
        )

        response_data = {
            "intent": "explanation",
            "answer": explanation.text
        }
    else:
        response_data = {
            "error": "Unable to determine request type"
        }

    conversation.append({
        "role": "assistant",
        "content": str(response_data)
    })

    return response_data