import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

df = pd.read_csv('loan_prediction.csv')

def create_data_summary(df):
    summary = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
    summary += "Columns:\n"
    for col in df.columns:
        summary += f"- {col} (type: {df[col].dtype})\n"
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
        model="gpt-4",
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