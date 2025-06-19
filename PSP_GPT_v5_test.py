import streamlit as st
import pandas as pd
import os

from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.agents import Tool

from langchain.chains import LLMChain

from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import AzureChatOpenAI



from langchain.agents import initialize_agent,AgentType

if 'df' not in st.session_state:
    st.session_state['df'] = ''

if 'cs' not in st.session_state:
    st.session_state['cs'] =''

if 'session' not in st.session_state:
    st.session_state['session'] =''


st.set_page_config(page_title="PSP AI Agent", layout="centered")
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





df = pd.read_excel('Gap2.xlsx')
df['Historical Date'] = pd.to_datetime(df['Historical Date'])
# Display the first few rows of the dataset

st.write('Preview of the uploaded file')
st.write(df.head(5))

@tool("KFT_tool", return_direct=True)
def KFT_tool(query: str) -> str:

    """
    Perform analysis on the `df` supplied and writes and executes Python code to give answers.

    Args:
        query (str): The search query.
        

    Returns:
        str: Result of the code execution (without extra commentary)
    """

    
    prompt_template = """
        You are an expert in writing Python code and executing it.
        
      ***Please answer only the current question asked by the user, without referencing previous questions. Follow the conditions provided in the current user input.***
        Question is :"{question}".
        I have the following CSV data with the columns: "{columns}". 
        Data is in a dataframe called 'df' already.
        Don't give any description, just write relevant and correct and error-free Python code and store output in a variable called result.
        Ignore the case in 'df' and also ignore case in the question the user asks.
        Please generate a Python script using this 'df' as input dataframe and pandas to answer this question: "{question}".
        Do not write any Python script which alters the dataframe 'df'.
        Write only the correct and error free code
         with exception handing read-only Python script and import streamlit as st. While using any column having Date values use 'dt' and not 'str'.
        Dont give any explanation while executing the python code.
        **Do not include any descriptions, explanations, or comments.**
        **Always respect the timeline given by user. Eg. if user is asking for H2, use second half of year etc.***
        STRICT REQUIREMENT:
        **Always start afresh for the new user query. Just remember the dataframe df and do not remember anything from the past written code**
        **Loop through all columns** that might be relevant for grouping (excluding `Historical Date`, `Gap`, `Plan_Data`, and `Amount (converted)`) and **group the data** by `Gap` one by one.
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
       
        
        The dataframe df contains multiple columns, each representing different entities (e.g., 2024 Legal Entity 1, 2024 Legal Entity 2, 2024 Legal Entity 3, 2024 Legal Entity 4).
        The LLM must automatically detect which column to use based on the user's input.
        If the user provides a location (e.g., "North America", "China", "India"), find the column that contains this value and use it for filtering and analysis.
        Example:
        If the user asks about "North America", and df["2024 Legal Entity 3"] contains "North America", then use df["2024 Legal Entity 3"] for further processing.
        If the user asks about "US" or "USA", and df["2024 Legal Entity 4"] contains "United States", then use df["2024 Legal Entity 4"].
    

        STRICTLY TO BE FOLLOWED:

        **Always respect the timeline given by user. Eg. if user is asking for H2, use second half of year etc.***

       ### Priority Hierarchy for Filtering and Grouping:
**For Location:**
    1. 2024 Legal Entity 1 (Highest)
    2. 2024 Legal Entity 2 (2nd Highest)
    3. 2024 Legal Entity 3 (3rd Highest)
    4. 2024 Legal Entity 4 (Lowest)

**For Product Line Code and Product Groups:**
    1. OBI Group (Highest)
    2. OBI Subgroup (2nd Highest)
    3. Secondary Sub-Group (3rd Highest)
    4. Product Line Code (Lowest)

**For Lead Source:**
    1. Grouped Lead Source (Highest)
    2. Lead Source (Lowest)

STRICTLY TO BE FOLLOWED:

- **Always and always remove the group that explains 100 percent of the Gap.**
- Whenever you filter the data for an entity in a hierarchy, then **do not include the same and higher elements in the hierarchy** for the grouping.
    - **For example, if the user asks for "Draw 1st level Pareto for United States," in the loop that you run involving other columns to find the group that explains the maximum gap, **exclude** `2024 Legal Entity 1`, `2024 Legal Entity 2`, and `2024 Legal Entity 3` from the grouping as `2024 Legal Entity 4` has the lowest priority in the hierarchy.**
    - Similarly, if you are filtering for "Product Line Code", in the loop to find the first-level Pareto, **exclude** 'OBI Group' for the grouping to be done for finding 1st level pareto.
    - **Similarly, if you are filtering for "Product Line Code", in the loop to find the first-level Pareto, **exclude** 'OBI Group'  , `OBI Sub Group`, `Secondary Sub-Group` (higher hierarchies) from the grouping,  and also remove `Product Line Code` (as filtrer is applied on it) in the loop for grouping.**
    - **Similarly, if you are filtering for "OBI Sub Group", in the loop to find the first-level Pareto, **exclude** `OBI Group` (highest priority) and `OBI Sub Group` (the filter applied) from the grouping, but **include** `Secondary Sub-Group` and `Product Line Code` in the loop for grouping.**
    - **Similarly, if you are filtering for "Lead Source", in the loop to find the first-level Pareto, **exclude** `Grouped Lead Source` (highest priority) and `Lead Source` (the filter applied) from the grouping.**
    - **Similarly, if you are filtering for "Grouped Lead Source", in the loop to find the first-level Pareto, **exclude** `Grouped Lead Source` (the filter applied) from the grouping but include `Lead Source` in the grouping as it's lower in hierarchy.**
### Strict Hierarchical Filtering for Pareto:
- The code must group and analyze the data only at the level specified by the user.


        
         Below is the mapping to be used if required
            User Input          - 2024 Legal Entity 3           - 2024 Legal Entity 4
            
           1. China	            - PRC China (W Suzhou)	        -''
           2. India	            - India and South West Asia	    -''
           3. US, USA             - North America                 - United States	 
           4. North America       - North America                 -''    
           5. Western Europe      -Western Europe                  -''
           6. LATAM               -Latin America                   -''
                         
        Another mapping:

            User Input          - 2024 Legal Entity 2           
            
           1. Greater Europe      - Greater Europe 
           2. America             - Americas LS
           
             

            E.g. If the input is India, you should consider 2024 Legal Entity 3
            and if the inout is US,USA, you should consider 2024 Legal Entity 4

            Also, if you cant find a direct mapping, look out for that column that has the user input.
            E.g if user input is for Australia, look out for that column that has Austrlaia
            as a part of the string in the column and then use that column.




        - **If the user asks to plot or draw a graph** (e.g., bar chart, line chart, etc.), without specifying any column, 
          plot the graph of each column vs `Gap` one by one by grouping each column against `Gap` 
          and use `matplotlib.pyplot` to generate the plot.
        - **If the user asks to plot or draw a graph**, use `matplotlib.pyplot` to generate the plot.
        - Identify the columns explicitly mentioned in the question.
        - Never ever group `Historical Date`,`Gap` by `Amount (converted)`, `Plan_Data`, even if the user asks.
        - Perform a **group by on these columns** except on the columns of `Plan` and `Amount (converted)`.
        - Take care of the below input from the user and its corresponding mapping in the `df` dataframe. Use your intelligence to infer the mappings in real-time: 
            - `High Geo` etc. in user input refers to column of `2024 Legal Entity 3` in df.
            - `Country`, `Nation` etc. in user input refers to column of`2024 Legal Entity 4` in df.
            - `Region` etc. in user input refers to column of `2024 Legal Entity 2` in df.
            - 'FFR' or 'Actual' refers to 'Amount (converted)' column in the dataframe.

        - Apply `sum()` to the `Gap` column.
        - `gap` or `GAP` refers to the column `Gap`.

       


        If the user asks to plot or draw a graph, use matplotlib.pyplot to generate the plot.
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
        
         **Loop through all columns** that might be relevant for grouping (excluding `Historical Date`, `Gap`, `Plan_Data`, and `Amount (converted)`) and **group the data** by `Gap` one by one.

        Group by each column individually (excluding Historical Date, Gap, Plan_Data, and Amount (converted)) on Gap.
         Remember the Hierarchy: 
        "2024 Legal Entity 1" is first, "2024 Legal Entity 2" is second, "2024 Legal Entity 3" is third while
        ""2024 Legal Entity 4" is last.
        So, if you are putting filter on "2024 Legal Entity 3" diregard  "2024 Legal Entity 1" , "2024 Legal Entity 2" for the grouping
        but consider  "2024 Legal Entity 4" for the grouping.

        Similarly,  if you are putting filter on "2024 Legal Entity 4" diregard  "2024 Legal Entity 1" , "2024 Legal Entity 2" 
        and "2024 Legal Entity 3" for the grouping.
        
        ### Strict Requirements:



        Sort each grouped category in descending order of the gap.
        Compute the sum of the top 3 elements within each grouped category.
        ***Identify the group whose top 3 elements have maximum sum of 'Gap'.
        Identify the second group whose top 3 elements have the second highest sum of 'Gap'.
        Identify the third  group whose top 3 elements have the third highest sum of 'Gap'.***
        Plot separate bar graphs for each of these 3 groups against Gap with sizeable gap bwteen the 3 graphs and put the 3 graphs
        vertically and mention the value of 'Gap' on top of each bar in the graph.


        Use figsize=(10, 25) if youa re drawing 3 graphs such that x labels dont overwrite the below graphs.

        # Use figsize=(5,5) if youa re drawing 3 graphs such that x labels dont overwrite the below graphs.

       
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
        and then find the third grouping top 3 of which have highest sum of 'Gap'
        

        1st Level Pareto Chart with Specific Columns:
        If the user specifies a column or multiple columns for the Pareto chart, follow these steps:

        *STRICT REQUIREMENT*:

        When user asks to draw '1st Level Pareto',
        **Loop through all columns** that might be relevant for grouping (excluding `Historical Date`, `Gap`, `Plan_Data`, and `Amount (converted)`) and **group the data** by `Gap` one by one
        after applying the filter the user asked.***

        ***When drawing graphs using "Product Line Code" in x axis, ALWAYS show the exact "Product Line Code" in the x-axis.
        E.g if in x-axis, we have Product Line Code 62 and 111, show the x-axis with the labels of 62 and 111.***
        ***If plotting a bar chart or any graph with x-axis labels, DO NOT rotate the labels by 90 degrees.***

        Filter the data based on the user's specification.
        Example: For "Draw 1st level Pareto of 'OBI Group' of 'Flow Cytometry' in 'North America'", filter the data for:
        'OBI Group' containing Flow Cytometry
        '2024 Legal Entity 3' containing North America
        Example: For "Draw 1st level Pareto of North America", filter the data to include only North America.
         **Loop through all columns** that might be relevant for grouping (excluding `Historical Date`, `Gap`, `Plan_Data`, and `Amount (converted)`) and **group the data** by `Gap` one by one.
        Group by each column individually (excluding Historical Date, Gap, Plan_Data, Amount (converted), and the user-specified columns) on Gap.
        Sort each grouped category in descending order of the gap.
        Compute the sum of the top 3 elements within each grouped category.
        Identify the first group whose top 3 elements have the highest sum of 'Gap'.
        Identify the second group whose top 3 elements have the second highest sum of 'Gap'.
        Identify the third  group whose top 3 elements have the third highest sum of 'Gap'.
        Plot separate bar graphs for each of these 3 groups against Gap and mention the value of 'Gap' on top of each bar in the graph.
        Example:

        Group by 2024 Legal Entity 3, Product Line Code, and other relevant columns on Gap.
        Sort each group by Gap in descending order.
        Identify the group where the top 3 elements have the highest sum of 'Gap'.
        Identify the second group where the top 3 elements have the second highest sum of 'Gap'.
        Identify the third  group where the top 3 elements have the third highest sum of 'Gap'.
        Plot separate bar graphs for each of these 3 groups against Gap.
        Strict Formatting Requirement:
        1.If plotting a bar chart or any graph with x-axis labels, DO NOT rotate the labels by 90 degrees.
        
        3.The response must strictly contain only the Python script—no explanations, headings, or inline comments.
        # 4. ***If the code produces error in output, rewrite the code. Try it till you get correct results or 5 times whichever happens first.***
        """


    # Dont give any description but
    # just the single python code for the question that I can use in python exec function. 
    # Don't include python and print in the text field of answer.
    # Create the PromptTemplate object using LangChaino
    template = PromptTemplate(
        input_variables=["columns", "question"],
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
                        "question": question
                        
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

@tool("code_writing_tool", return_direct=True)
def code_writing_tool(query: str, num_results: int = 5) -> str:
    """
    Only used when user asks expilcitly to write a code for something else doesnt get used

    Args:
        query (str): The input query by user for writing code.
        
    Returns:
        str: Code written in the language user asks
        
    """

    def analyze_query_with_ai(query: str) -> bool:
        """
        Use AI to determine if crawling is needed based on the query.
        """
        prompt = (
            f"Decide if the following search query requires writing code or not: "
            f"'{query}'. Respond with 'yes' if it requires to write code in any programming language else reply with 'no."
        )
        try:
  
           
        
            messages = [
                {"role": "system", "content": "You are an intelligent assistant that can decide if user is asking you to write code."},
                {"role": "user", "content": prompt}
            ]
            
            # Invoke the LLM (replace with your LLM API invocation)
            response = llm.invoke(messages)
            decision = response.content

            return decision == "yes"
        except Exception as e:
            print(f"AI analysis failed: {e}")
            return False

    # Use AI to determine if crawling is required
    write_code = analyze_query_with_ai(query)

    if(write_code == 'yes'):
            prompt = f"Based on the following {query},  write the code for {query}"
            template = PromptTemplate(
                        input_variables=["query"],
                        template=prompt,
                    )

                    # Create the LLMChain to manage the model and prompt interaction
            llm_chain = LLMChain(prompt=template, llm=llm)
            response = llm_chain.invoke({
                "content" : query
            })      
            
            
            return response["text"]



code_tool = Tool(
    name="Code Writer",
    func=code_writing_tool,
    description="A tool for writing code."
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

    tools=[KFT_finding_tool]

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

    with st.chat_message("assistant"):
        # st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages)
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)

