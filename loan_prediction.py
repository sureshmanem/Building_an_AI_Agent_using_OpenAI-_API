import pandas as pd
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Check if using Azure OpenAI or regular OpenAI
use_azure = os.getenv('USE_AZURE_OPENAI', 'false').lower() == 'true'

if use_azure:
    # Azure OpenAI configuration
    client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
    )
    model_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4')
    print("Using Azure OpenAI Service")
else:
    # Regular OpenAI configuration
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    model_name = os.getenv('OPENAI_MODEL', 'gpt-4')
    print("Using OpenAI Service")

df = pd.read_csv('loan_prediction.csv')

def create_data_summary(df):
    summary = f"Dataset Overview:\n"
    summary += f"- Total Records: {df.shape[0]}\n"
    summary += f"- Total Columns: {df.shape[1]}\n\n"
    
    summary += "Detailed Column Information:\n"
    summary += "=" * 60 + "\n"
    
    for col in df.columns:
        summary += f"\n{col}:\n"
        summary += f"  Data Type: {df[col].dtype}\n"
        
        # Missing values
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        summary += f"  Missing Values: {missing_count} ({missing_pct:.1f}%)\n"
        
        # Numerical columns
        if df[col].dtype in ['int64', 'float64']:
            summary += f"  Statistics:\n"
            summary += f"    - Mean: {df[col].mean():.2f}\n"
            summary += f"    - Median: {df[col].median():.2f}\n"
            summary += f"    - Min: {df[col].min():.2f}\n"
            summary += f"    - Max: {df[col].max():.2f}\n"
            summary += f"    - Std Dev: {df[col].std():.2f}\n"
        
        # Categorical columns
        else:
            unique_count = df[col].nunique()
            summary += f"  Unique Values: {unique_count}\n"
            
            if unique_count <= 10:  # Show distribution for low cardinality columns
                value_counts = df[col].value_counts()
                summary += f"  Value Distribution:\n"
                for val, count in value_counts.items():
                    pct = (count / len(df)) * 100
                    summary += f"    - {val}: {count} ({pct:.1f}%)\n"
            else:
                summary += f"  Top 5 Values: {', '.join(map(str, df[col].value_counts().head(5).index.tolist()))}\n"
    
    summary += "\n" + "=" * 60 + "\n"
    return summary  

def agent_ai(user_query, df):
    data_context = create_data_summary(df)

    prompt = f"""
        You are a data expert AI agent.

        You have been provided with this dataset summary:
        {data_context}

        Now, based on the user's question:
        '{user_query}'

        Think step-by-step. Assume you can access and analyze the dataset like a Data Scientist would using Pandas.

        Give a clear, final answer.
    """
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    return answer

print("Welcome to Loan Review AI Agent!")
print("You can ask anything about the loan applicants data.")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nYour question: ")
    if user_input.lower() == "exit":
        break
    response = agent_ai(user_input, df)
    print("\nAI Agent Response:")
    print(response)