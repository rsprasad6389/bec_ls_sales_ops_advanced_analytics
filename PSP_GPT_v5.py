import streamlit as st
import pandas as pd
import os
# import ollama
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.agents import Tool

from langchain.chains import LLMChain
# from langchain_ollama import ChatOllama
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import AzureChatOpenAI
# import snowflake.connector as sf
import re
from datetime import datetime
from typing import Tuple
from openai import OpenAI  # or AzureOpenAI, or use LangChain LLM interface
import pandas as pd

from langchain.agents import AgentExecutor

flag_column = 0
user_specific_value = 0
# from snowflake.snowpark import Session
from langchain.agents import initialize_agent,AgentType

if 'df' not in st.session_state:
    st.session_state['df'] = ''

if 'cs' not in st.session_state:
    st.session_state['cs'] =''

if 'session' not in st.session_state:
    st.session_state['session'] =''



list_matching_cols = []
st.set_page_config(page_title="PSP AI Agent", layout="wide")
result = ''

llm = AzureChatOpenAI(
    api_key = "082d3990364b4fadbc133fa8935b7905",
                        azure_endpoint = "https://becopenaidev7.openai.azure.com/",
                        model = "gpt-4o",
                        api_version="2024-02-01",
                        temperature = 0.
    # other params...
)



# Create a LangChain LLM using the Hugging Face pipeline
os.environ['SSL_CERT_FILE'] = 'C:\\Users\\RSPRASAD\\AppData\\Local\\.certifi\\cacert.pem'
# GROQ_API_KEY = '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'
# llm = ChatOllama(model = 'codellama', temperature = 0)



df = pd.read_excel('Gap2.xlsx', sheet_name = 'Gap2')

df.rename(columns={'2024 Legal Entity 4': 'Country'}, inplace=True)

st.write('Preview of the uploaded file')
st.write(df.head(5))



df['Historical Date'] = pd.to_datetime(df['Historical Date'])
# Display the first few rows of the dataset

df = df[df['Historical Date'].dt.month<=5]



import re
from typing import Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import dateparser  # install via: pip install dateparser

def interpret_time_range_with_nlp(query: str):
    """
    Use LLM and NLP to interpret natural language date ranges like 
    'between March and May 2025', 'from Jan to April', etc.

    Returns either (start_date, end_date) or filtered df.
    """
    prompt = f"""
You are a smart date parsing assistant. Convert the following human query into a time period.
Return a brief natural sentence indicating the start and end, like:
"From March 2025 to May 2025" or "Before April 2024".

Do NOT write full date strings unless obvious — we will parse them using NLP later.

Time query: {query}
"""
    global df
    response = llm.invoke(prompt)
    response = response.content.strip()
    st.write(f'response is..{response}')

    
    # Find all date-like expressions using NLP
    possible_dates = re.findall(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[\s\-]\d{4})?\b|\b\d{4}-\d{2}-\d{2}\b", response, re.IGNORECASE)

    parsed_dates = [dateparser.parse(d, settings={"PREFER_DAY_OF_MONTH": "first"}) for d in possible_dates]
    parsed_dates = [d for d in parsed_dates if d is not None]

    parsed_dates = sorted(parsed_dates)
    start_date = parsed_dates[0] if parsed_dates else None
    end_date = parsed_dates[-1] if len(parsed_dates) > 1 else None

    # Optional filtering
    if df is not None and 'Historical Date' in df.columns:

        st.write(f'Starting period is..{start_date}')
        st.write(f'Ending period is..{end_date}')         
        df['Historical Date'] = pd.to_datetime(df['Historical Date'])

        if start_date and end_date:
            df=  df[(df['Historical Date'] >= start_date) & (df['Historical Date'] <= end_date)]
        elif start_date:
            df =  df[df['Historical Date'] >= start_date]
        elif end_date:
           df= df[df['Historical Date'] <= end_date]
        else:
           df = df.copy(deep = True)

        st.write(max(df['Historical Date']))
        st.write(min(df['Historical Date']))

    return start_date, end_date

def extract_keywords(query):
    """Extract words in single or double quotes."""
    raw_matches = re.findall(r"'(.*?)'|\"(.*?)\"", query)
    # Flatten the list of tuples and remove empty strings
    return [kw.strip().lower() for tup in raw_matches for kw in tup if kw]

column_hierarchy = [
   
    "2024 Legal Entity 1",
    "2024 Legal Entity 2",
    "2024 Legal Entity 3",
    "Country",
    "Grouped Lead Source",
    "Lead Source",
    "OBI Group",
    "OBI Sub Group",
    "Secondary Sub-Group",
    "Product Line Code"
]

cols_to_group = ''

def find_matching_columns(keyword, keywords):
    """Find columns where the keyword appears (case-insensitive)."""

    global df  # if df is defined globally
    global list_matching_cols
    global cols_to_group
    keyword_lower = keyword.lower()
    global flag_column 
    global df
    global user_specific_value 

    for col in df.columns: 
        
        if keyword_lower == col.lower():
            for keyword in keywords:
                if df[col].astype(str).str.contains(keyword, case=False, na=False).any():
                    st.write(f'Applying the user given filter of {keyword} in column of {col}')
                    mask = df[col].astype(str).str.contains(keyword, case=False, na=False)
                    df = df[mask].copy(deep = True)
                    user_specific_value = 1
                    flag_column = 0
                    # return col
            
            st.write('Exact match for column has been found')
            st.write(f'Col is {col}')
            st.write('Type of col is', str(type(col)))
            cols_to_group = [col]
            st.write(f'Cols to group is {cols_to_group}')
            list_matching_cols = [col]
            st.write(f'list_matching_cols is {list_matching_cols}')
            flag_column = 1
            return col

    for col in column_hierarchy:

        # if col in df.columns:
 
        # First check: keyword in column name
    
        flag_column = 0
        if keyword_lower in col.lower():
            cols_to_group = list(col)
            flag_column = 1
            return col
            
        # if col in df.columns:
        if df[col].astype(str).str.contains(keyword, case=False, na=False).any():
            return col  # Return only the first match based on hierarchy

    return None  # No match found

def apply_filters(query):
    keywords = extract_keywords(query)
    global df
    
    st.write('I am in apply filters..')
    st.write(keywords)
    for keyword in keywords:
        global list_matching_cols
        matching_col = find_matching_columns(keyword, keywords)
        list_matching_cols.append(matching_col)
        st.write(list_matching_cols)
        st.write(matching_col)
        if not matching_col:
            print(f"❌ No matching columns found for keyword: '{keyword}'")
            return 0  # Empty DF if any keyword is unmatched

        if flag_column ==0:
            mask = df[matching_col].astype(str).str.contains(keyword, case=False, na=False)
            st.write('Applying filter of ', matching_col, 'in df')
            df = df[mask].copy(deep = True)
        
        st.write(df.head())
        st.write('flag_column is..', flag_column)


def fun_cols_to_group():
    st.write('I am in cols_to_group..')
    st.write('flag_column is..', flag_column)
    global cols_to_group
    global  list_matching_cols
    cols_to_group = [col for col in df.columns if col not in [
        'Historical Date', 'KPI', 'YEAR', 'Gap', 'Plan_Data', 'Amount (converted)', '2024 Legal Entity 1'
    ]]

    # Define removal rules


    if(flag_column ==0 or user_specific_value ==1):
        if '2024 Legal Entity 2' in list_matching_cols:
            cols_to_group = [col for col in cols_to_group if col != '2024 Legal Entity 1']

        if '2024 Legal Entity 3' in list_matching_cols:
            cols_to_group = [col for col in cols_to_group if col not in ['2024 Legal Entity 1', '2024 Legal Entity 2']]

        if 'Country' in list_matching_cols:
            cols_to_group = [col for col in cols_to_group if col not in ['2024 Legal Entity 1', '2024 Legal Entity 2', '2024 Legal Entity 3']]

        if 'OBI Sub Group' in list_matching_cols:
            cols_to_group = [col for col in cols_to_group if col != 'OBI Group']

        if 'Secondary Sub-Group' in list_matching_cols:
            cols_to_group = [col for col in cols_to_group if col not in ['OBI Group', 'OBI Sub Group']]

        if 'Lead Source' in list_matching_cols:
            cols_to_group = [col for col in cols_to_group if col != 'Grouped Lead Source']

        cols_to_group = [col for col in cols_to_group if col not in list_matching_cols]

    if(flag_column ==1):
        cols_to_group = list_matching_cols

@tool("KFT_tool", return_direct=True)
def KFT_tool(query: str) -> str:
    


    """
    Perform analysis on the `df` supplied and writes and executes Python code to give answers.

    Args:
        query (str): The search query.
        

    Returns:
        str: Result of the code execution (without extra commentary)
    """
    global df
    interpret_time_range_with_nlp(query)
    apply_filters(query)
    # st.write('I am going in cols_to_group..')
    fun_cols_to_group()
    st.write(f'Columns to group are {cols_to_group}')
    # st.write()
   
    # df = pd.read_csv(StringIO(df))
    
    prompt_template = """
        You are an expert in writing Python code and executing it.
        

        ***IMPORTANT: Under no circumstance should the dataframe 'df' be filtered or subsetted.***
        ***Never write code like df[...] or df[df['col'] == value]. Always use the entire 'df' as-is.***
        ***The dataframe `df` is already pre-processed and must be used **as-is**.
        ***If any filtering is needed (e.g., H2 or Japan), it has already been applied — do not do it again.***
        Question is :"{question}".
        I have the following CSV data with the columns: "{columns}" and list of 'list_matching_cols' is :"{list_matching_cols}"
        and list of `cols_to_group` is "{cols_to_group}"
        Data is in a dataframe called 'df' already.
        ***Strict Filtering Rule:***
        **Always and always remove the group that explains 100 percent of the Gap.**
        If the global list "{list_matching_cols}" has values, don't put any filter on the dataframe.
        **Absolutely do not filter the dataframe 'df' under any condition. Do not create any filtered version of df such as filtered_df. Always use 'df' as-is.**

        Don't give any description, just write relevant and correct and error-free Python code and store output in a variable called result.
        Ignore the case in 'df' and also ignore case in the question the user asks.
        Please generate a Python script using this 'df' as input dataframe and pandas to answer this question: "{question}".
        Do not write any Python script which alters the dataframe 'df'.
        Write only the correct and error free code
         with exception handing read-only Python script and import streamlit as st. While using any column having Date values use 'dt' and not 'str'.
        Dont give any explanation while executing the python code.
        **Do not include any descriptions, explanations, or comments.**

        STRICT REQUIREMENT:

       1. **Always start afresh for the new user query. Just remember the dataframe df and do not remember anything from the past written code**
      Hierarchically group and filter the DataFrame using these columns.

    
    




        2.**group the `Gap` column by elements of the `cols_to_group` on `Gap` one by one using a loop.

        
         

        
        Sort elements in each group in descending order of `Gap`.
        if there is just one element in any grouped category, remove that grouped category from your consideration.
        Compute the sum of the top 3 elements within each grouped category and remove other elements from the grouped category
        ***Identify the group whose top 3 elements have maximum sum of 'Gap'. Plot the graph for that group first.
        Identify the second group whose top 3 elements have the second highest sum of 'Gap'.Plot the graph for that group next.
        Identify the third  group whose top 3 elements have the third highest sum of 'Gap'.***Plot the graph for that group next.

        Sort each grouped category in descending order of the gap.
        Plot separate bar graphs for each of these 3 groups against Gap with sizeable gap bwteen the 3 graphs and put the 3 graphs
        vertically and mention the value of 'Gap' on top of each bar in the graph.


        Use figsize=(10, 25) if youa re drawing 3 graphs such that x labels dont overwrite the below graphs.

        # Use figsize=(5,5) if youa re drawing 3 graphs such that x labels dont overwrite the below graphs.

        While plotting the graphs, show just 3 bar graphs within each graph

        Also on top of each graph, print the sum of top 3 elements of the graph

       
        Example:

        Group by 2024 Legal Entity 3, Product Line Code, and other relevant columns on Gap.
        Sort each group by Gap in descending order.
        Identify the group where the sum of 'Gap' of top 3 elements is the maximum.
        
        Identify the second group whose top 3 elements have the second highest sum of 'Gap'.
        Identify the third  group whose top 3 elements have the third highest sum of 'Gap'.
        Plot separate bar graphs for each of these 3 groups against Gap.

        So, imagine you group Gap by '2024 Legal Entity 3', 'Product Line Code', 'OBI Group' etc and then sort each group in descending order of 'Gap'.
        Imagine sum of top 3 elements of 'OBI Group' when sorted in descending order of 'Gap' is maximumn i.e, greater than the 
        sum of top 3 elements of 'Product Line Code' when sorted in descending order of 'Gap' 
        and greater than  sum of top 3 elements of '2024 Legal Entity 3' when sorted in descending order of 'Gap' etc 

        then select the grouping of 'Gap' by  'OBI Group', then find the second grouping top 3 of which have highest sum of 'Gap'
        and then find the third grouping top 3 of which have highest sum of 'Gap' and plot grouping of 'Gap' by  'OBI Group'first then
        grouping of 'Gap' by  'Product Line Code' and then  grouping of 'Gap' by  '2024 Legal Entity 3'
                


        When drawing graphs using "Product Line Code" on the x-axis, ALWAYS use the exact "Product Line Code" values from the dataframe as x-axis labels.
        Do not modify, abbreviate, concatenate, or alter them in any way.
        Ensure that the x-axis values appear exactly as they are in the dataframe.
        If "Product Line Code" contains numeric values, explicitly convert them to strings before plotting to prevent Matplotlib from applying automatic formatting.
        Example:
        python
        Copy
        Edit
        df["Product Line Code"] = df["Product Line Code"].astype(str)

        **Show only 3 bars in the bar Graph from the left.**
       
        

        **Always respect the timeline given by user. Eg. if user is asking for H2, use second half of year etc.***

     



**If the user asks to plot or draw a graph**, use `matplotlib.pyplot` to generate the plot.



        
       

        Ensure that you use st.pyplot(fig) to display the graph instead of plt.show().
        Before plotting, set the figure size using fig = plt.figure(figsize=(5,5)).
        When plotting, arrange the largest gap on the leftmost side, followed by the second largest, and so on.


       
        Display the y-axis in thousands of dollars ($ K).
        1st Level Pareto Chart:
        If the user requests a 1st level Pareto chart, follow these steps:
        *STRICT REQUIREMENT*:
        ***When drawing graphs using "Product Line Code" in x axis, ALWAYS show the exact "Product Line Code" in the x-axis.
        E.g if in x-axis, we have Product Line Code 62 and 111, show the x-axis with the labels of 62 and 111.***
        ***If plotting a bar chart or any graph with x-axis labels, DO NOT rotate the labels by 90 degrees.***
        
       

        

       """


    # Dont give any description but
    # just the single python code for the question that I can use in python exec function. 
    # Don't include python and print in the text field of answer.
    # Create the PromptTemplate object using LangChaino
    template = PromptTemplate(
        input_variables=["columns", "question", "list_matching_cols", "cols_to_group"],
        template=prompt_template,
    )
    # When asked about plotting a graph, use st.pyplot(fig) where fig = plt.figure(figsize=(10,10)) 
            # While drawing the graphs, put x axis labels rotated by 90.    
    # # Create the LLMChain to manage the model and prompt interaction
    llm_chain = LLMChain(prompt=template, llm=llm)



    # ---------------------------
    # Streamlit App Interface
    # ---------------------------









    # Ask the user for a question about the data
    question = query
    # while True:
    if question:
        with st.spinner("Generating Python code..."):
            attempt = 0
            global result
            while attempt <5:
                try:
                    
                    # Run the LLMChain to generate the Python script based on the question and CSV columns
                    python_script = llm_chain.invoke({
                        "columns": ", ".join(df.columns),
                        "question": question,
                        "list_matching_cols": list_matching_cols, 
                        "cols_to_group" : cols_to_group
                        
                    })

                    # Display the generated Python code
                    # st.write(df.head())
                    # st.write("### 📝 Generated Python Code:")
                    python_script['text'] = python_script['text'].strip('`').replace('python', '')
                    st.code(python_script['text'], language='python')
                    # st.write('Executing the code to give the answers')
                    # Option to execute the generated Python code
                    # if st.button("▶️ Run Code"):
                    try:
                    
                        import matplotlib.pyplot as plt
                        

                        exec_locals = {}
                        exec_globals = {"df": df, "pd": pd}
                        # st.write(python_script)
                        # st.write(python_script['text'].strip('`').replace('python', ''))

                        python_script['text'] = python_script['text'].strip('`').replace('python', '')
                        exec_globals = globals().copy()  # Start with all existing globals

                        # Append additional required objects
                        exec_globals.update({
                            "plt": plt
                            
                        })
                        
                        exec(python_script['text'], exec_globals, exec_locals)

                        # st.write(exec(python_script['text'], {'df':df, 'pd':pd}))
                    
                    
                        # If a result variable is present, display it
                        if 'result' in exec_locals:
                            st.write("### 📊 Result:")
                            st.write(exec_locals['result'])
                            st.stop()
                            return(str(exec_locals['result']))
                            break
                            
                        else:
                            st.warning("⚠️ The code did not produce a 'result' variable.")
                            st.stop()
                    except Exception as e:
                        st.error(f"🚫 Error running the code 1: {e}")
                        # st.stop()
                        attempt += 1
                        # return ('Error')
                except Exception as e:
                    st.error(f"🚫 Error generating the code 2: {e}")
                    attempt += 1
                    # return('Error')

KFT_finding_tool= Tool(
    name="KFT Tool",
    func=KFT_tool,
    description="A tool for answering questions based on the user query"
)


if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot that can do PSP analysis"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    # search=DDGS(verify = False).text(prompt, max_results=10) 
    # llm=ChatGroq(groq_api_key=GROQ_API_KEY,model_name="Llama3-8b-8192",streaming=True)
    
    # tools=[KFT_finding_tool, code_tool]

    # tools=[filter_tool,KFT_finding_tool]
    tools = [KFT_tool]

    llm = AzureChatOpenAI(
    api_key = "082d3990364b4fadbc133fa8935b7905",
                        azure_endpoint = "https://becopenaidev7.openai.azure.com/",
                        model = "gpt-4o",
                        api_version="2024-02-01",
                        temperature = 0.
    # other params...
)


    search_agent=initialize_agent(tools,llm,
                                  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                  handle_parsing_errors=True, verbose = False)
    

    
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=search_agent,
        tools=[KFT_finding_tool],
        max_iterations=40,      # ← increase this
        max_execution_time=120,  # ← or this (in seconds)
    )

    with st.chat_message("assistant"):
        # st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        # response=search_agent.run(st.session_state.messages)
        st.write(prompt)
        response=search_agent.run(prompt)
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)
        # st.write(matched_cols)
