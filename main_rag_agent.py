import asyncio
from typing import TypedDict
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.base import StructuredTool
from cric_rag_agent_t20i import t20i_qna_agent
from cric_rag_agent_ipl import ipl_qna_agent
from cric_rag_agent_psl import psl_qna_agent
from cric_rag_agent_odi import odi_qna_agent
from cric_rag_agent_bpl import bpl_qna_agent
from cric_rag_agent_bbl import bbl_qna_agent
from cric_rag_agent_ilt import ilt20_qna_agent
from cric_rag_agent_cpl import cpl_qna_agent
from cric_rag_agent_ntb import ntb_qna_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import os
import dotenv   
import time
import streamlit as st

dotenv.load_dotenv()  # Load environment variables from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

async def multi_rag_agent(input):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    prev_state = {"chat_history": st.session_state.chat_history}

    class ConversationAgent(TypedDict):
        input: str
        agent_outcome: object  # AgentAction / AgentFinish
        intermediate_results: list[str]
        tool_names: list[str]
        tool_details: list[dict]
        tool_executed: bool
        generation: str
        chat_history: list[str]
        direct_answer: str

    t20i_tool = StructuredTool.from_function(coroutine=t20i_qna_agent)
    ipl_tool = StructuredTool.from_function(coroutine=ipl_qna_agent)
    psl_tool = StructuredTool.from_function(coroutine=psl_qna_agent)
    odi_tool = StructuredTool.from_function(coroutine=odi_qna_agent)
    bpl_tool = StructuredTool.from_function(coroutine=bpl_qna_agent)
    bbl_tool = StructuredTool.from_function(coroutine=bbl_qna_agent)
    cpl_tool = StructuredTool.from_function(coroutine=cpl_qna_agent)
    ilt20_tool = StructuredTool.from_function(coroutine=ilt20_qna_agent)
    ntb_tool = StructuredTool.from_function(coroutine=ntb_qna_agent)

    t20i_tool.handle_tool_error = True
    ipl_tool.handle_tool_error = True
    psl_tool.handle_tool_error = True
    odi_tool.handle_tool_error = True
    bpl_tool.handle_tool_error = True
    bbl_tool.handle_tool_error = True
    cpl_tool.handle_tool_error = True
    ilt20_tool.handle_tool_error = True
    ntb_tool.handle_tool_error = True

    tools = {"t20i_qna_agent": t20i_tool, "ipl_qna_agent": ipl_tool, "psl_qna_agent": psl_tool, "odi_qna_agent": odi_tool,"bpl_qna_agent": bpl_tool,"bbl_qna_agent": bbl_tool,"cpl_qna_agent":cpl_tool,"ilt20_qna_agent":ilt20_tool,"ntb_qna_agent": ntb_tool}

    llm = ChatOpenAI(temperature=0, model="gpt-4.1-2025-04-14", max_tokens=500)

    main_agent_prompt = ChatPromptTemplate.from_messages(
    [
            ("system", "You are an advanced conversational Q&A AI assistant designed to answer user queries using available tools and provided files. Your responses must be precise, efficient, and strictly based on tool outputs whenever the question is about cricket data, stats, or matches. For general knowledge or conversational queries (e.g., 'Who is Virat Kohli?', 'Who are you?'), answer directly in a natural, conversational way without relying on tools. Always maintain a helpful, friendly, and professional tone. Execution Process: Tool Selection & Execution: Identify and execute the appropriate tool(s) when the query is about cricket stats, matches, tournaments, or performance data. Use multiple tools if necessary. For non-cricket or conversational questions, respond directly without tools. Avoid speculation; rely strictly on tool outputs for cricket stats. All the files are preloaded inside their corresponding tools as per the file format. Always provide tools with the necessary context to generate accurate responses. Use the user's previous interactions to maintain context continuity and generate a proper query if it lacks context before invoking tools. When user queries contain pronouns or incomplete references (e.g., 'his top run in T20I'), resolve them using entities from previous chat history (e.g., Virat Kohli from last turn). Always rewrite the query into a complete and explicit form before invoking a tool. Never pass pronouns or incomplete references directly. When resolving ambiguous references, prefer the most recent relevant entity from chat history. Before invoking a tool, verify that the rewritten query is fully self-contained and unambiguous. If query is ambiguous and cannot be resolved confidently from chat history, request clarification from the user. For general or conversational queries (like greetings, identity, or biographical questions), do not use tools and do not mention missing data. Answer directly, naturally, and clearly as a conversational assistant."
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
    ]
    )

    llm_with_tools = main_agent_prompt | llm.bind_tools(tools.values())

    tool_name_map = {"IPL-T20": "ipl_qna_agent","T20I": "t20i_qna_agent", "PSL": "psl_qna_agent", "ODI": "odi_qna_agent","BPL": "bpl_qna_agent","BBL":"bbl_qna_agent","CPL":"cpl_qna_agent","ILT": "ilt20_qna_agent","NTB": "ntb_qna_agent"}


    def extract_tool_queries(agent_outcome):
        tool_queries = []

        # Handle additional_kwargs
        tool_calls = agent_outcome.additional_kwargs.get("tool_calls", [])

        for call in tool_calls:
            name = call["function"]["name"]
            args_str = call["function"].get("arguments", "{}")

            # Parse arguments safely
            import json
            try:
                args = json.loads(args_str)
            except Exception:
                args = {}

            tool_queries.append({
                "tool": name,
                "query": args.get("user_query", "")
            })

        return tool_queries

    async def main_agent(data):
        response = await llm_with_tools.ainvoke(data)

        tool_details = extract_tool_queries(response)

        if not tool_details:
            return {
                "agent_outcome": response,
                "tool_details": [],
                "direct_answer": True
            }

        return {"agent_outcome": response, "tool_details": tool_details, "direct_answer": False}

    async def execute_tools(data):

        agent_action = data['tool_details'] 
        outputs = []

        for detail in agent_action:
            tool_name = detail["tool"]
            tool_query = detail.get("query", data["input"])  

            internal_tool_name = tool_name_map.get(tool_name, tool_name)

            if internal_tool_name in tools:
                print(f"Invoking tool: {internal_tool_name} with input: {tool_query}")
                time.sleep(0.5)
                result = await tools[internal_tool_name].coroutine(user_query=tool_query)
                outputs.append(result)
            else:
                outputs.append(f"Unknown tool: {tool_name}")

        return {
            "intermediate_results": outputs,
            "tool_executed": "True"
        }

    async def generate_answer(data):
            """
            Generate answer from retrieved_results using llm
            """
            question = data["input"]
            retrieved_results = data["intermediate_results"]

            template = """
            Your job is to produce a **complete, accurate, and human-friendly answer**.

            ---

            ### Instructions

            #### 1. Understand the User Query
            * Identify exactly what the user is asking.
            * Focus only on the information that directly addresses the question.
            * If the user says anything like "hi", "hello" respond like "hi, how can i help you today?"

            #### 2. Use All Retrieved Data
            * Carefully review all retrieved results.
            * **Include every single data point provided by the sub-agents.**
            * Do not skip, omit, summarize, or filter out any part of the results.
            * If multiple results are provided, **merge and present them all** in a logical order.

            #### 3. Generate a Clear, Human-Readable Answer
            * Present the information in **natural language**.
            * You may reformat for readability (e.g., bullet points, numbered lists, tables, or short paragraphs).
            * Formatting must improve clarity, but the **content must remain 100 percent intact and complete**.
            * Clearly format numerical data, player stats, and match summaries.

            #### 4. No Raw or Technical Output
            * Never output plain SQL results, raw lists, or JSON.
            * Rewrite the results into a **well-phrased, coherent explanation** — but without losing or altering any detail.

            #### 5. Handle Missing or Incomplete Data
            * If results are incomplete, explicitly state:
            *“Only partial data is available for this query.”*
            * Never invent or hallucinate missing details.

            ---

            ### Input
            * **User Question:** `{question}`
            * **Retrieved Results:** `{retrieved_results}`

            ---

            ### Output Requirement
            * A **complete, structured, and human-like answer** that:
            * Directly answers the user's question.
            * **Contains every single piece of information provided by the sub-agents.**
            * Uses human-readable formatting (bullets, tables, or prose).
            * Is free from technical jargon and database formatting.

            """

            prompt = PromptTemplate(input_variables=["question","retrieved_results"],template=template)

            rag_chain = (prompt| llm | StrOutputParser())

            result = await rag_chain.ainvoke({"question":question,"retrieved_results":retrieved_results})

            return {"generation": result}

    def should_continue(data):
        if data.get("direct_answer"):
            return "end"            
        if data['tool_executed'] == "True":
            return "final_answer"
        return "continue"


    # --------------------- build workflow --------------------- #
    workflow = StateGraph(ConversationAgent)
    workflow.add_node("agent", main_agent)
    workflow.add_node("action", execute_tools)
    workflow.add_node("final_answer", generate_answer)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "action")
    workflow.add_conditional_edges("action", should_continue, {"continue": "agent","final_answer": "final_answer","end": END})
    workflow.add_edge("final_answer", END)
    workflow = workflow.compile()
    output = await workflow.ainvoke({"input": input, "chat_history": prev_state["chat_history"]})

    st.session_state.chat_history.append(HumanMessage(content=input))

    if "generation" in output:
        reply = output["generation"]
    elif isinstance(output.get("agent_outcome"), str):
        reply = output["agent_outcome"]
    else:
        reply = getattr(output.get("agent_outcome"), "content", str(output.get("agent_outcome")))

    st.session_state.chat_history.append(AIMessage(content=reply))
    return reply



# if __name__ == "__main__":
#     query = "hi"
#     result = asyncio.run(multi_rag_agent(query))
#     print(result)


"""
"Who scored the most runs in IPL 2023?"

"Top 2 wicket-takers in IPL 2020?"

"Which team won the IPL 2022 final and who was the Man of the Match?"

"Who is the leading run scorer in ICC world cup in 2022?"

"Compare Virat Kohli's best scores in IPL 2023 and ICC world cup 2022."

"Between Jos Buttler's IPL 2022 season and his T20 World Cup 2022 performance, which was better in terms of runs scored?"
"""
