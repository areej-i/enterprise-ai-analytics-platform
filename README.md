# Enterprise AI Analytics Platform

An end-to-end AI-powered analytics platform that allows users to upload datasets and documents, ask questions in natural language, 
perform machine learning tasks, generate visualizations, and retrieve information from uploaded documents using Retrieval-Augmented Generation (RAG).
The application combines LLMs, traditional data analytics, machine learning, RAG, FastAPI, Docker, and CI/CD into a single application.

---

## Features

### AI-Powered Data Analysis

Users can upload CSV datasets and ask questions about their data using natural language. The application uses an LLM to determine the user's intent 
and translate the request into a structured execution request.

Supported analysis operations include:

* Mean
* Sum
* Minimum
* Maximum
* Median
* Count
* Missing-value analysis
* Unique-value analysis
* Grouped analysis

---

### 📊 Data Visualization

The platform can generate visualizations based on natural-language requests. 

Supported chart types:

* Bar charts
* Line charts
* Scatter plots
* Histograms
* Box plots

Note that a CSV file must be uploaded beforehand.
---

### Machine Learning Engine

The platform supports several machine learning workflows through a dedicated ML engine.

#### Classification
Used when predicting categorical outcomes.

Example:
> "Can you predict whether an employee will leave the company based on the other columns?"

The ML engine:
* Preprocesses numerical and categorical features
* Handles missing values
* Encodes categorical variables
* Splits data into training and testing sets
* Trains a Random Forest classifier
* Calculates evaluation metrics
* Calculates feature importance
* Saves the trained model

Metrics include:
* Accuracy
* Precision
* Recall
* F1 Score

#### Regression

Used when predicting numerical values.

Example:
> "Can you predict employee salary based on the other features?"

Metrics include:
* MAE
* RMSE
* R²

#### Clustering
Used to identify groups within unlabeled data.

Example:
> "Can you group these employees based on their characteristics?"

The platform uses clustering techniques to identify groups and evaluate the resulting clusters.

---

### 🔎 Retrieval-Augmented Generation (RAG)

Users can upload PDF documents and ask questions about their contents.

The RAG pipeline:

1. Accepts an uploaded PDF
2. Extracts the document text
3. Splits the document into chunks
4. Generates embeddings
5. Stores embeddings in a FAISS vector database
6. Retrieves relevant chunks when a question is asked
7. Provides the retrieved context to the LLM
8. Generates an answer grounded in the retrieved document

Note: One limitation is the user must explicitly refer to the document in their prompt. For example:
According to the uploaded document, what is overfitting?
What does the document say about the vacation policy?

---

### 📁 File Uploads

The FastAPI backend supports user uploads for:

* CSV datasets
* PDF documents

Uploaded CSV files become the active dataset for analysis and machine learning.

Uploaded PDFs are processed through the RAG pipeline.

---


## Architecture

```text
                         ┌──────────────────┐
                         │      Client      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     FastAPI      │
                         │      /ask        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   Gemini LLM     │
                         │  Intent Router   │
                         └────────┬─────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │    Analysis  │  │      ML      │  │     RAG      │
        │    Engine    │  │    Engine    │  │    Engine    │
        └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
               │                 │                 │
               ▼                 ▼                 ▼
            Pandas           Scikit-learn        FAISS
               │                 │                 │
               ▼                 ▼                 ▼
          Calculations       ML Results       Retrieved
          & Charts           & Models          Context
```

---

## Technology Stack

### AI / NLP

* Python
* Google Gemini
* Hugging Face / local embedding models
* LangChain

### Data Science / Machine Learning

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

### RAG

* LangChain
* FAISS
* PDF document processing
* Local embedding model

### Backend

* FastAPI
* Pydantic
* Uvicorn

### DevOps

* Docker
* Git
* GitHub
* GitHub Actions
* AWS
* Amazon ECR

### Testing

* Pytest

---

## API Endpoints

### Ask a Question

```http
POST /ask
```

Example request:

```json
{
  "question": "What department has the highest average salary?"
}
```

The LLM determines the appropriate intent and routes the request to the corresponding engine.

---

### Upload a CSV

```http
POST /upload/csv
```

Uploads a CSV dataset and initializes the analysis and machine learning engines using the uploaded data.

Example response:

```json
{
  "message": "CSV uploaded successfully.",
  "filename": "employees.csv",
  "rows": 1000,
  "columns": [
    "name",
    "salary",
    "department"
  ]
}
```

---

### Upload a PDF

```http
POST /upload/pdf
```

Uploads and processes a PDF document for RAG.

The document is:

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS Vector Database
```

---

## Example Questions

### Data Analysis

```text
What is the average salary?
Which department has the highest salary?
How many employees are in the dataset?
Show me the distribution of salaries.
What are the unique departments?
```

### Visualization

```text
Create a bar chart of average salary by department.
Create a histogram of salaries.
Show me a scatter plot of salary versus age.
Create a box plot of salaries.
```

### Machine Learning

```text
Build a classification model to predict employee attrition.
Predict salary using regression.
What are the most important features in the classification model?
Cluster the employees into groups.
How well did the regression model perform?
```

### RAG

```text
What does data normalization mean?
What is the difference between supervised and unsupervised learning?
According to the uploaded document, what is overfitting?
```

---

## Machine Learning Pipeline

The ML engine automatically handles preprocessing for numerical and categorical data.

```text
Uploaded Dataset
       │
       ▼
Select Target
       │
       ▼
Separate Features / Target
       │
       ▼
Data Preprocessing
 ┌─────┴─────┐
 ▼           ▼
Numeric   Categorical
 ▼           ▼
Imputation Encoding
 └─────┬─────┘
       ▼
   Train/Test Split
       │
       ▼
   ML Algorithm
       │
       ▼
   Evaluation
       │
       ├── Metrics
       ├── Feature Importance
       └── Saved Model
```

---

## RAG Pipeline

```text
             PDF Upload
                  │
                  ▼
          Document Processing
                  │
                  ▼
              Chunking
                  │
                  ▼
             Embeddings
                  │
                  ▼
            FAISS Vector DB
                  │
                  │
User Question ────┤
                  ▼
          Similarity Search
                  │
                  ▼
         Relevant Document Chunks
                  │
                  ▼
               Gemini
                  │
                  ▼
              Answer
```

---

## Testing

The project includes automated tests using Pytest.

Run the test suite with:

```bash
python -m pytest
```

The tests currently cover functionality such as:

* Mean calculations
* Aggregations
* Counts
* Grouped calculations
* Invalid column handling
* Error handling

---

## CI/CD

GitHub Actions is used to automatically run the test suite when changes are pushed to the repository.

The CI pipeline verifies that changes do not break existing functionality.

The project also includes a Docker-based deployment workflow with **Amazon ECR** for storing container images.

```text
Git Push
   │
   ▼
GitHub Actions
   │
   ├── Install dependencies
   ├── Run tests
   └── Build Docker image
           │
           ▼
      Amazon ECR
```

---

## Docker

Build the Docker image:

```bash
docker build -t enterprise-ai-analytics .
```

Run the container:

```bash
docker run -p 8000:8000 enterprise-ai-analytics
```

The FastAPI API can then be accessed through:

```text
http://localhost:8000
```

FastAPI's interactive API documentation is available at:

```text
http://localhost:8000/docs
```

---

## Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
LLM_MODEL=gemini-2.5-flash
```

Do **not** commit your `.env` file to GitHub.

Add it to `.gitignore`:

```text
.env
.venv/
__pycache__/
uploads/
```

---

## Running Locally

Clone the repository:

```bash
git clone https://github.com/areej-i/enterprise-ai-analytics-platform.git
```

Enter the project:

```bash
cd enterprise-ai-analytics-platform
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your environment variables:

```text
.env
```

Run the FastAPI application:

```bash
uvicorn api:app --reload
```

Open:

```text
http://localhost:8000/docs
```

---

## Design Principles

The project separates responsibilities between the LLM and deterministic Python components.

### LLM

The LLM is responsible for:

* Understanding natural-language requests
* Determining user intent
* Selecting analysis or ML operations
* Explaining results
* Generating RAG responses

### Python

Python is responsible for:

* Numerical calculations
* Data preprocessing
* Machine learning
* Model evaluation
* Feature importance
* Visualization
* Document processing
* Vector search

This prevents the LLM from being responsible for calculations that can be performed more reliably using deterministic tools.

---

## Future Improvements

Potential future improvements include:

* Multi-user dataset sessions
* Persistent conversation storage
* Authentication
* Cloud-hosted vector databases
* Streaming responses
* More advanced ML algorithms
* Automated model selection
* Hyperparameter tuning
* Forecasting
* Cloud deployment
* Frontend interface
* Monitoring and logging

---

## Author

**Areej Irfan**

Computer Science graduate focused on **AI, machine learning, NLP, data science, and cloud technologies**.
