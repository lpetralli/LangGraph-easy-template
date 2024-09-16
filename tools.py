from langchain.tools.retriever import create_retriever_tool

def create_retriever_tool_from_vectorstore(vectorstore):
    retriever = vectorstore.as_retriever()
    return create_retriever_tool(
        retriever,
        "retrieve_company_docs",
        "Search and return information about the company documents",
    )

from typing import Dict
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

class ClientInfoInput(BaseModel):
    client_id: str = Field(..., description="The unique identifier of the client")

class ClientInfoOutput(BaseModel):
    name: str
    email: str
    account_balance: float
    installed_products: list
    sustainability_score: float

class GetClientInfoTool(BaseTool):
    name = "get_client_info"
    description = "Retrieves client information for TechnoVerde S.A. customers based on the provided client ID"
    args_schema = ClientInfoInput

    def _run(self, client_id: str) -> Dict:
        # Mocked client data for TechnoVerde S.A.
        mock_client_data = {
            "TV001": {
                "name": "María González",
                "email": "maria@ecohome.com",
                "account_balance": 2500.0,
                "installed_products": ["Solar Panels", "Smart Thermostat"],
                "sustainability_score": 8.5
            },
            "TV002": {
                "name": "Carlos Rodríguez",
                "email": "carlos@greenbusiness.com",
                "account_balance": 10000.0,
                "installed_products": ["Energy Management System", "LED Lighting"],
                "sustainability_score": 9.2
            },
            "TV003": {
                "name": "Ana Martínez",
                "email": "ana@sustainablefuture.org",
                "account_balance": 5000.0,
                "installed_products": ["Water Conservation System", "Electric Vehicle Charger"],
                "sustainability_score": 7.8
            },
        }

        if client_id not in mock_client_data:
            raise ValueError(f"TechnoVerde S.A. client with ID {client_id} not found")

        client_info = mock_client_data[client_id]
        return ClientInfoOutput(**client_info).dict()

def create_get_client_info_tool():
    return GetClientInfoTool()
