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
            PII_PROMPT_TEMPLATE.format_messages(messages=run.inputs["messages"])
        )
        return EvaluationResult(
            **{"key": "pii", "comment": result.reasoning, "score": result.contains_pii}
        )

TOPIC_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at analyzing conversations and identifying unanswered questions or unaddressed topics. Your task is to carefully compare the user's input with the AI's response and determine if any part of the user's query was not adequately addressed."),
    ("human", """Analyze the following conversation:

User Input: {input}

AI Response: {output}

Determine if there's any part of the user's input that wasn't adequately addressed in the AI response. Focus on:
1. Specific questions left unanswered
2. Topics or subjects mentioned by the user but not discussed in the response
3. Requests for information that were not fulfilled
""")
])

class TopicResult(BaseModel):
    new_topic_detected: bool = Field(..., description="Whether a new, unaddressed topic was detected")
    explanation: str = Field(..., description="Explanation of the topic analysis")

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
            return EvaluationResult(key="new_topic", score=None)
        
        result = self.evaluator.with_structured_output(TopicResult, method='json_schema').invoke(
            TOPIC_PROMPT_TEMPLATE.format_messages(input=run.inputs["messages"], output=run.outputs["messages"])
        )
        
        return EvaluationResult(
            key="new_topic",
            comment=result.explanation,
            score=float(result.new_topic_detected)
        )
