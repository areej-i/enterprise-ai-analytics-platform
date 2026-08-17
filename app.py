import pandas as pd
from dotenv import load_dotenv
from gemini_service import GeminiService
from analysis_engine import AnalysisEngine
from ml_engine import MachineLearningEngine
from document_processor import DocumentProcessor
from rag_engine import RAGEngine


df = pd.read_csv("datasets/employees.csv")

# csv_path = input("Enter the path to your CSV file: ")
# df = pd.read_csv(csv_path)

df.columns = df.columns.str.lower()


gemini = GeminiService()
analysisEngine = AnalysisEngine(df)
MLEngine = MachineLearningEngine(df)
processor = DocumentProcessor()
rag_engine = RAGEngine()


load_dotenv()

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

dataset_profile = create_dataset_profile(df)

conversation = []
conversation_summary = ""

while True:
    
    user_prompt = input("Ask Gemini something. To exit, type 'exit': ")
    
    # If the user types "exit", break the loop
    if user_prompt.lower() == "exit":
        break

    # If the conversation is too long, summarize it and keep only the last 10 messages
    # if len(conversation) > 20:

    #     conversation_summary = gemini.summarize_conversation(
    #         conversation
    #     )

    #     conversation = conversation[-10:]

    if len(conversation) > 5:

        conversation = conversation[-3:]
    

    conversation.append(
    {
        "role": "user",
        "content": user_prompt
    }
    )

    #Gemini decides the operation
    request = gemini.create_analysis_request(
        user_prompt,
        dataset_profile,
        # conversation_summary,
        conversation
    )

    print("Gemini requested:")
    print(request)
    print()
    print()

    # If user wants an explanation
    if request.intent == "analysis":

        result = analysisEngine.execute(request)

        print(result)
        print()
        print()

        response_text = explain_result(
            user_prompt,
            result
        )

    elif request.intent == "machine_learning":

        if request.task in ["classification", "regression"]:
            if not request.target_column:
                response_text = (
                    "Classification and regression require a target column."
                )

            elif request.target_column not in df.columns:
                response_text = (
                    f"Target column '{request.target_column}' does not exist. "
                    f"Available columns: {list(df.columns)}"
                )

            else:
                result = MLEngine.execute(request)

                response_text = explain_ML_result(
                    user_prompt,
                    result
                )

        elif request.task == "clustering":
            result = MLEngine.execute(request)
            response_text = explain_ML_result(
                user_prompt,
                result
            )

        else:
            response_text = ("I don't know how to perform that machine learning task yet.")

    # If user wants an explanation
    elif request.intent == "explanation":

        explanation = gemini.generate(
            f"""
            You are a data analyst assistant.

            Use the dataset information below to answer
            the user's question.

            Dataset Information:
            {dataset_profile}

            User Question:
            {user_prompt}
            """
        )

        response_text = explanation.text

    elif request.intent == "rag":
        context = rag_engine.retrieve_context(user_prompt)

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
            {user_prompt}
            """
        )

        response_text = explanation.text
    
    else:
        response_text = (
            "I'm not sure how to answer that question yet."
        )


    print(response_text)
    print()
    print()

    conversation.append(
    {
        "role": "assistant",
        "content": response_text
    }
)