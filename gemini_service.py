import os
from urllib import response
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schemas import ExecutionRequest

load_dotenv()

class GeminiService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = os.getenv(
            "LLM_MODEL",
            "gemini-2.5-flash"
        )

    # 
    def generate(self, prompt, response_schema=None, conversation=None):

        config = None

        if response_schema:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=conversation if conversation else prompt,
            config=config
        )

        return response

    def summarize_conversation(self, conversation):

        prompt = f"""

        Summarize this conversation.

        Keep important:
        - user goals
        - questions asked
        - conclusions
        - important dataset findings

        Conversation:

        {conversation}

        """

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return response.text

    def create_analysis_request(
            self,
            question,
            dataset_profile,
            # conversation_summary,
            conversation
        ):

        conversation_text = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in conversation
            ]
        )

        prompt = f"""
        You are an AI data analyst that plans how a Python application should answer user questions.

        Dataset Information:
        {dataset_profile}

        Previous Conversation:
        {conversation_text}

        Your job is to classify the user's request into ONE of the following intents.

        -------------------------------------------------------
        INTENT 1: explanation

        Use this when NO dataframe calculations or machine learning are required.

        Examples:
        - What columns are in the dataset?
        - What does the salary column mean?
        - What is a histogram?
        - Explain correlation.
        - Explain the previous result.
        - What did you mean by that?

        Do NOT use explanation if any data must be calculated.

        -------------------------------------------------------
        INTENT 2: analysis

        Use this when the answer can be produced using standard dataframe operations.

        This includes:
        - mean
        - median
        - min
        - max
        - sum
        - count
        - missing values
        - unique values
        - chart generation

        Examples:
        - What is the average salary?
        - Which department has the highest salary?
        - Show a histogram of salaries.
        - Count the employees.
        - Which country has the most customers?

        If the user requests a chart/visualization:

        Histogram
        - x_column = numeric column
        - y_column = null

        Box Plot
        - x_column = null
        - y_column = numeric column

        Scatter Plot
        - x_column = independent variable
        - y_column = dependent variable

        Ex:
        "Plot age versus salary"
        x_column = age
        y_column = salary

        Line Chart
        - group_by = time or ordered variable when applicable
        - value_column = measured value

        Bar Chart
        - group_by = categories
        - value_column = values to plot

        Do NOT use analysis if the user is asking for predictions, classification, regression, clustering, forecasting, or model training.

        -------------------------------------------------------
        INTENT 3: machine_learning

        Use this when the user wants the application to build, train, evaluate, or use a machine learning model.

        Examples:

        Classification
        - Predict employee attrition.
        - Train a model to classify spam.
        - Build a churn prediction model.

        Regression
        - Predict salaries.
        - Predict house prices.
        - Predict future revenue.

        Clustering
        - Group similar customers.
        - Cluster employees by behaviour.
        - Segment users.

        Forecasting
        - Forecast sales.
        - Predict next month's revenue.

        Model evaluation
        - Train a model.
        - Evaluate the model.
        - Show feature importance.

        -------------------------------------------------------
        INTENT 4: rag

        Use only when the user is asking about uploaded documents, PDFs, company policies,
        knowledge base articles, manuals, or documentation. The user does NOT need to mention 
        "PDF" explicitly but does need to reference an uploaded document.

        Do not use this intent for dataframe analysis.

        Examples:
        - "According to the document, what is exploratory data analysis?"
        - "What does the uploaded material say about regression?"
        - "Summarize the section about hypothesis testing"
        - "Find information about visualization techniques from the documents"

        -------------------------------------------------------
        User Question:

        {question}

        Return ONLY the fields relevant to the selected intent.

        If intent = explanation:
        - operation = null
        - x_column = null
        - y_column = null
        - chart_type = null
        - task = null
        - target_column = null

        If intent = analysis:
        - Fill operation/x_column/y_column/chart_type only if needed.
        - task = null
        - target_column = null

        If intent = machine_learning:
        - operation = null
        - x_column = null
        - y_column = null
        - chart_type = null
        - Fill task.
        - Fill target_column if the task requires one.

        Do not perform the calculation yourself.
        Only decide how the application should handle the request.

        For machine learning tasks:

        - target_column MUST exactly match one of the dataset column names.
        - Never create a new column name.
        - If the user describes a concept instead of a column name, find the closest matching column from the dataset information.
        - Return the exact spelling of the column.
        """

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExecutionRequest
            )
        )


        return response.parsed