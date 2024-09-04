from typing import Optional
from langsmith.evaluation import RunEvaluator, EvaluationResult
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


PII_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at identifying Personally Identifiable Information (PII) in text. Your task is to carefully analyze the provided messages and determine if they contain any PII."),
    ("human", "Please analyze the following messages for any PII:\n{messages}")
])

class PIIResult(BaseModel):
    contains_pii: bool = Field(..., description="Whether the message contains PII")
    reasoning: str = Field(..., description="Explanation of the PII evaluation")

class PIIEvaluator(RunEvaluator):
    
    def __init__(self):
        self.evaluator = ChatOpenAI(model="gpt-4o-2024-08-06")

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        if (
            not run.inputs
            or not run.inputs.get("messages")
            or not run.outputs
            or not run.outputs.get("messages")
        ):
            return EvaluationResult(key="pii", score=None)
        result = self.evaluator.with_structured_output(PIIResult, method='json_schema').invoke(
            PII_PROMPT_TEMPLATE.format_messages(messages=run.inputs["messages"][-1])
        )
        return EvaluationResult(
            **{"key": "pii", "comment": result.reasoning, "score": result.contains_pii}
        )

TOPIC_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at analyzing conversations and identifying when requested information is not available. Your task is to carefully compare the user's input with the AI's response and determine if the AI indicated that the requested information was not available in its knowledge base or documents."),
    ("human", """Analyze the following conversation:

User Input: {input}

AI Response: {output}

Determine if the AI's response indicates that the requested information was not available in its knowledge base or documents. Focus on:
1. Explicit statements about lack of information
2. Indications that the AI cannot answer due to limited data
3. Responses suggesting the information is beyond the AI's current knowledge
""")
])

class TopicResult(BaseModel):
    information_unavailable: bool = Field(..., description="Whether the AI indicated that the requested information was not available")
    explanation: str = Field(..., description="Explanation of the analysis")

class TopicEvaluator(RunEvaluator):
    
    def __init__(self):
        self.evaluator = ChatOpenAI(model="gpt-4o-2024-08-06")

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        if (
            not run.inputs
            or not run.inputs.get("messages")
            or not run.outputs
            or not run.outputs.get("messages")
        ):
            return EvaluationResult(key="information_unavailable", score=None)
        
        result = self.evaluator.with_structured_output(TopicResult, method='json_schema').invoke(
            TOPIC_PROMPT_TEMPLATE.format_messages(input=run.inputs["messages"][-1], output=run.outputs["messages"][-1])
        )
        
        return EvaluationResult(
            key="information_unavailable",
            comment=result.explanation,
            score=float(result.information_unavailable)
        )
