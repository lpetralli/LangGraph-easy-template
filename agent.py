from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from LangGraph import build_graph
\
from langchain_core.tracers import EvaluatorCallbackHandler

from typing import Optional
from langchain.evaluation import load_evaluator
from langsmith.evaluation import RunEvaluator, EvaluationResult
from langsmith.schemas import Run, Example

from dotenv import load_dotenv

load_dotenv()

from evaluators import PIIEvaluator
pii_callback = EvaluatorCallbackHandler(evaluators=[PIIEvaluator()])

from evaluators import TopicEvaluator
topic_callback = EvaluatorCallbackHandler(evaluators=[TopicEvaluator()])

class Agent:
    def __init__(self, model_type="openai", prompt=None, tools=None):
        if model_type == "openai":
            self.model = ChatOpenAI(temperature=0, model_name="gpt-4o")
        elif model_type == "groq":
            self.model = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
        else:
            raise ValueError("Unsupported model type. Please choose 'openai' or 'groq'.")
        
        self.tools = tools if tools else []
        self.system_prompt = prompt
        
        # Build the graph using the function build_graph
        self.graph = build_graph(llm=self.model, prompt=self.system_prompt, tools=self.tools)

    def invoke(self, messages):
    
        # Initialize the state with the provided messages
        initial_state = {"messages": messages}

        # Run the graph synchronously and obtain the output
        graph_output = self.graph.invoke(initial_state, {"callbacks":[pii_callback, topic_callback]})
        
        return graph_output
