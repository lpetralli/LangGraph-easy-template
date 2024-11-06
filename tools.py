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



from typing import Literal, Optional, Union, List

recommendations = {
    "weather": {
        "warm": {
            "header": "Warm Destinations",
            "image": "https://img.freepik.com/free-photo/beautiful-tropical-beach-sea-with-coconut-palm-tree-paradise-island_74190-2206.jpg",
            "description": "Sunny beaches and tropical paradises for those seeking warmth and relaxation"
        },
        "cold": {
            "header": "Cold Destinations",
            "image": "https://img.freepik.com/free-photo/romantic-portrait-woman-white-dress-sailing-large-boat-ferry_343596-2643.jpg?t=st=1728570886~exp=1728574486~hmac=c8789ae592df118e04d779026e7e787d9f8f2026509940e5a267fae84d615125&w=1380",
            "description": "Snowy landscapes and winter wonderlands for adventure seekers and snow enthusiasts"
        }
    },
    "destination": {
        "beach": {
            "header": "Beach Getaway",
            "image": "https://img.freepik.com/free-photo/empty-sea-beach-background_74190-313.jpg?t=st=1728570972~exp=1728574572~hmac=3b36753278983aaf2ced5309d7106c9d2c8b941ddb0ca7d80e4cf23558acb1c9&w=1380",
            "description": "Relaxing beach destination for a peaceful getaway"
        },
        "mountain": {
            "header": "Mountain Retreat",
            "image": "https://img.freepik.com/free-photo/breathtaking-view-snowy-mountains-cloudy-sky-patagonia-chile_181624-9696.jpg?t=st=1728571010~exp=1728574610~hmac=347392b917bdd215ea2887333d224b92c0f242b003bf631f4e59160ad9aff551&w=1380",
            "description": "Exciting mountain destination with breathtaking views"
        }
    },
    "activities": {
        "with": {
            "header": "Adventure Activities",
            "image": "https://img.freepik.com/free-photo/girls-looking-something-forest_23-2147617377.jpg?t=st=1728571051~exp=1728574651~hmac=dbf4be11c9f520c3b49e38510df9d3bb0fc8c6ecefa1abb49af94d18ba995172&w=1380   ",
            "description": "Destinations with plenty of exciting activities"
        },
        "without": {
            "header": "Relaxation",
            "image": "https://img.freepik.com/free-photo/young-man-relax-bed-enjoying-mountain-view_1423-236.jpg?t=st=1728571077~exp=1728574677~hmac=f7e6fac257df211d06cfcdff793e9bd3522215d5245392871b2e05892283f07a&w=1380",
            "description": "Peaceful destinations for ultimate relaxation"
        }
    }
}


class ShowRecommendationsInput(BaseModel):
    recommendation_type: Literal["weather", "destination", "activities"] = Field(..., description="The type of recommendation to show.")

class ShowRecommendationsTool(BaseTool):
    name = "show_recommendations"
    description = "Use this tool to show travel recommendations based on the specified type: 'weather', 'destination', or 'activities'."
    args_schema = ShowRecommendationsInput

    def _run(self, recommendation_type: str) -> str:
        if recommendation_type not in recommendations:
            return f"No recommendations available for {recommendation_type}"

        options = recommendations[recommendation_type]
        result = f"Recommendations for {recommendation_type}:\n\n"

        for option_key, option_data in options.items():
            result += f"- {option_data['header']}:\n"
            #result += f"  Image: {option_data['image']}\n"
            result += f"  Description: {option_data['description']}\n\n"

        return result

def create_show_recommendations_tool():
    return ShowRecommendationsTool()

# Filter Itineraries

itineraries = [
    {
        "id": 1,
        "name": "Sunny Caribbean Cruise",
        "weather": "tropical",
        "destination": "beach",
        "activities": ["relaxation", "adventure"],
        "budget": "standard",
        "image": "https://img.freepik.com/free-photo/beautiful-asian-female-woman-relax-casual-leisure-peaceful-moment-cruise-deck-vacation-summer-time_609648-740.jpg?t=st=1728586030~exp=1728589630~hmac=5f9d775b140ff0e3945436a71ea99916e102878a5ef28e967804b4fe761145f7&w=1060",
        "description": "Enjoy the warm sun ☀️ and beautiful beaches 🏖️ of the Caribbean."
    },
    {
        "id": 2,
        "name": "Alaskan Glacier Expedition",
        "weather": "polar",
        "destination": "mountain",
        "activities": ["wildlife", "adventure"],
        "budget": "premium",
        "image": "https://img.freepik.com/free-photo/couple-traveling-together-country-side_23-2149406524.jpg?t=st=1728586064~exp=1728589664~hmac=91a0dd915456e9f9c0b0471bc884da8c992af7deae017298736e1070fb346f75&w=1060",
        "description": "Explore the rugged beauty of Alaska's glaciers ❄️🏔️."
    },
    {
        "id": 3,
        "name": "Mediterranean Cultural Tour",
        "weather": "temperate",
        "destination": "city",
        "activities": ["cultural", "relaxation"],
        "budget": "standard",
        "image": "https://img.freepik.com/free-photo/mother-her-daughter-eating-harvested-olives-field_23-2147907340.jpg?t=st=1728586103~exp=1728589703~hmac=fc21a3c94e3b206e2718b7bb9bdd265fb46e377f56a92c1963940f8df8d45f17&w=1060",
        "description": "Discover the rich history 🏛️ and culture 🎭 of the Mediterranean."
    },
    {
        "id": 4,
        "name": "African Safari Adventure",
        "weather": "tropical",
        "destination": "countryside",
        "activities": ["wildlife", "adventure"],
        "budget": "premium",
        "image": "https://img.freepik.com/free-photo/beautiful-shot-group-african-wildebeests-grassy-plain_181624-27243.jpg?t=st=1728586136~exp=1728589736~hmac=feb958d54b32c4181738e8ee8a31daeb5d4f2e8e1c1d935e59875df122f2b54e&w=1060",
        "description": "Experience the thrill of a safari 🦁🐘 in the African savannah."
    },
    {
        "id": 5,
        "name": "Japanese Cherry Blossom Tour",
        "weather": "temperate",
        "destination": "city",
        "activities": ["cultural", "family"],
        "budget": "standard",
        "image": "https://img.freepik.com/free-photo/row-cherry-blossom-tree-with-cherry-blossom-falling-petals-springtime-kyoto-japan_335224-1334.jpg?t=st=1728586179~exp=1728589779~hmac=c4a18674195aac1738342937ba82b1a521ad05822b8d625f879a728915fd6e45&w=1060",
        "description": "Witness the beauty of cherry blossoms 🌸 in Japan's historic cities."
    },
    {
        "id": 6,
        "name": "Skiing in the Swiss Alps",
        "weather": "temperate",
        "destination": "mountain",
        "activities": ["sports", "relaxation"],
        "budget": "luxury",
        "image": "https://img.freepik.com/free-photo/beautiful-view-people-cycling-skiing-across-snowy-mountains-south-tyrol-dolomites-italy_181624-29926.jpg?t=st=1728586203~exp=1728589803~hmac=2c0ebcbabe122750cd22a3abd175eb4918eb3d4964391cc9e651ebf9355013b1&w=1060",
        "description": "Hit the slopes ⛷️ and enjoy the scenic beauty of the Swiss Alps 🏔️."
    },
    {
        "id": 7,
        "name": "Romantic Getaway in Paris",
        "weather": "temperate",
        "destination": "city",
        "activities": ["romantic", "cultural"],
        "budget": "premium",
        "image": "https://img.freepik.com/free-photo/couple-browsing-smartphones-date_23-2147744393.jpg?t=st=1728586228~exp=1728589828~hmac=c5fc8c4b7ee9e2712329f6aa0b5eab5c921f7d16313b6eac2d7f22155611cbbd&w=1060",
        "description": "Spend a romantic weekend exploring the City of Love ❤️🗼."
    },
    {
        "id": 8,
        "name": "Wellness Retreat in Bali",
        "weather": "tropical",
        "destination": "island",
        "activities": ["wellness", "relaxation"],
        "budget": "luxury",
        "image": "https://img.freepik.com/free-photo/young-woman-with-body-positive-appearance-practicing-yoga-alone-deck-by-pool-tropical-island-bali-indonesia-sport-fitness-healthy-lifestyle-concept_1321-2876.jpg?t=st=1728586249~exp=1728589849~hmac=d7de6f2a422e314ee77e4d1d13e8446f237dc0e03d03e194b607793c562a18c4&w=1060",
        "description": "Rejuvenate your body and soul 🧘‍♀️ at a luxurious Bali retreat."
    },
    {
        "id": 9,
        "name": "Antarctic Expedition Cruise",
        "weather": "polar",
        "destination": "island",
        "activities": ["adventure", "wildlife"],
        "budget": "ultra_luxury",
        "image": "https://img.freepik.com/free-photo/couple-traveling-together-country-side_23-2149406534.jpg?t=st=1728586313~exp=1728589913~hmac=789a525758617383757f7a1b447b8bd99383e0e483b0723fcfe539d71de91e17&w=1060",
        "description": "Explore the icy wilderness of Antarctica 🐧❄️ on an expedition cruise."
    },
    {
        "id": 10,
        "name": "Family Fun at Disney World",
        "weather": "temperate",
        "destination": "city",
        "activities": ["family", "entertainment"],
        "budget": "standard",
        "image": "https://img.freepik.com/free-photo/full-shot-friends-posing-funfair_23-2148618877.jpg?t=st=1728586340~exp=1728589940~hmac=5c972dd60628aaf5095d6b9f98795b17b35c35b9247547cc3708fca24d9db560&w=1060",
        "description": "Enjoy magical moments with the whole family 👨‍👩‍👧‍👦🏰 at Disney World."
    },
    {
        "id": 11,
        "name": "Island Hopping in Greece",
        "weather": "temperate",
        "destination": "island",
        "activities": ["relaxation", "cultural"],
        "budget": "standard",
        "image": "https://img.freepik.com/free-photo/curly-short-haired-woman-floral-dress-boater-runs-outside_197531-24118.jpg?t=st=1728586367~exp=1728589967~hmac=166f34846ffc74a648e343b1c3cdaedee78d946ac6ba2f6d7f27402299e17096&w=1060",
        "description": "Discover the beauty of Greek islands 🏝️ and their rich history 🏛️."
    },
    {
        "id": 12,
        "name": "Australian Outback Adventure",
        "weather": "temperate",
        "destination": "countryside",
        "activities": ["adventure", "wildlife"],
        "budget": "standard",
        "image": "https://img.freepik.com/free-photo/sideways-woman-man-waving-each-other-coast_23-2148699842.jpg?t=st=1728586391~exp=1728589991~hmac=4eeb54923ae0c8f006e5605a67ec541dc0bf3c463286afadd2e57d7460b2f95c&w=1060",
        "description": "Experience the rugged terrain and unique wildlife 🦘🐨 of the Outback."
    },
    {
        "id": 13,
        "name": "Luxury Nile River Cruise",
        "weather": "tropical",
        "destination": "city",
        "activities": ["cultural", "relaxation"],
        "budget": "luxury",
        "image": "https://img.freepik.com/free-photo/dubai-creek_158595-1992.jpg?t=st=1728586419~exp=1728590019~hmac=6bba3729c093988d6f0145b1218e8ab81769f8893502c59585f8ff4ada5fbe51&w=1060",
        "description": "Sail along the Nile 🚢 and explore ancient Egyptian wonders 🐪🏺."
    },
    {
        "id": 14,
        "name": "Wellness Spa in the Himalayas",
        "weather": "temperate",
        "destination": "mountain",
        "activities": ["wellness", "relaxation"],
        "budget": "premium",
        "image": "https://img.freepik.com/free-photo/person-practicing-cold-exposure-metabolism_23-2150981869.jpg?t=st=1728586439~exp=1728590039~hmac=fa05e3f31d865cd61a73b94024d3154820de644e1101b383d72d11e1ae483b82&w=1060",
        "description": "Find peace at a spa retreat 🧘‍♂️ nestled in the Himalayas 🏔️."
    },
    {
        "id": 15,
        "name": "Wine Tasting in Tuscany",
        "weather": "temperate",
        "destination": "countryside",
        "activities": ["cultural", "relaxation"],
        "budget": "premium",
        "image": "https://img.freepik.com/free-photo/low-angle-happy-friends-partying-outdoors_23-2149412443.jpg?t=st=1728586466~exp=1728590066~hmac=41ea894916a7aa2cde3abe55e96722eb1d2b50514552128803f0d14b7090daf7&w=1060",
        "description": "Indulge in fine wines 🍷 and picturesque landscapes 🌄 in Tuscany."
    },
    {
        "id": 16,
        "name": "Exploring the Amazon Rainforest",
        "weather": "tropical",
        "destination": "countryside",
        "activities": ["adventure", "wildlife"],
        "budget": "standard",
        "image": "https://img.freepik.com/free-photo/young-traveler_1150-5651.jpg?t=st=1728586488~exp=1728590088~hmac=da09e8b99f9b4176272fedb77ff1e44b16b19aec133f06a7e7b7665c9e3920be&w=1060",
        "description": "Dive into the heart of the Amazon 🌴 and its diverse ecosystem 🐒🦜."
    },
    {
        "id": 17,
        "name": "Northern Lights in Iceland",
        "weather": "polar",
        "destination": "countryside",
        "activities": ["adventure", "romantic"],
        "budget": "premium",
        "image": "https://img.freepik.com/free-photo/beautiful-aurora-borealis-sky-iceland-spectacular-green-violet-northern-lights-appearing-night-creating-panoramic-landscape-glowing-magical-natural-phenomenon-starry-sky_482257-69775.jpg?t=st=1728586508~exp=1728590108~hmac=bd9c9198ed1575ccd2f2e5de5765d519ae0e4fcc72cfb6b748117b2536f2d038&w=1060",
        "description": "Witness the breathtaking Northern Lights 🌠 in Iceland."
    },
    {
        "id": 18,
        "name": "Yoga Retreat in Costa Rica",
        "weather": "tropical",
        "destination": "beach",
        "activities": ["wellness", "relaxation"],
        "budget": "standard",
        "image": "https://img.freepik.com/free-photo/side-view-woman-doing-yoga-nature-with-copy-space_23-2148769597.jpg?t=st=1728586527~exp=1728590127~hmac=f8024676c2f6afd27fe310c854a170835ea6edd2d2bbd7af02dc839efc52975a&w=1060",
        "description": "Rebalance with yoga sessions 🧘‍♀️ on Costa Rica's serene beaches 🏖️."
    },
    {
        "id": 19,
        "name": "Historical Tour of Rome",
        "weather": "temperate",
        "destination": "city",
        "activities": ["cultural", "family"],
        "budget": "standard",
        "image": "https://img.freepik.com/free-photo/couple-honeymoon-venice_1303-5723.jpg?t=st=1728586552~exp=1728590152~hmac=e84cdb3be4cdeed6828a13f38d7fe20bb89e4d6f93f913e3424ad1df5bfefd65&w=1060",
        "description": "Explore ancient ruins 🏛️ and art 🎨 in the heart of Rome."
    },
    {
        "id": 20,
        "name": "Beach Party in Ibiza",
        "weather": "temperate",
        "destination": "beach",
        "activities": ["entertainment", "sports"],
        "budget": "premium",
        "image": "https://img.freepik.com/free-photo/medium-shot-friends-partying-outdoors_23-2149646131.jpg?t=st=1728586575~exp=1728590175~hmac=5944ce04045e762ef353cd594dc332ea88de42251f248a3b7fc43123557c9fd9&w=1060",
        "description": "Enjoy vibrant nightlife 🎉 and water sports 🏄‍♂️ on Ibiza's beaches."
    },
    {
        "id": 21,
        "name": "Cycling Tour of the Netherlands",
        "weather": "temperate",
        "destination": "countryside",
        "activities": ["sports", "cultural"],
        "budget": "economy",
        "image": "https://img.freepik.com/free-photo/transport-concept-with-people-bicycles_23-2148959676.jpg?t=st=1728586602~exp=1728590202~hmac=caef57fc727d3e91ca3a97a9e5f55184959886f8c891c03b6187eee3aa906bac&w=1060",
        "description": "Cycle through picturesque landscapes 🚲 and historic towns 🏘️."
    },
    {
        "id": 22,
        "name": "Wildlife Expedition in Madagascar",
        "weather": "tropical",
        "destination": "island",
        "activities": ["wildlife", "adventure"],
        "budget": "standard",
        "image": "https://img.freepik.com/free-photo/beautiful-cheetah-standing-big-branch_181624-18632.jpg?t=st=1728586632~exp=1728590232~hmac=b65cefeac7a230ed709c0b1c5f2b2f09bbaa0fe55247801a2f5548d9f11cf473&w=1060",
        "description": "Discover unique species 🦎 on an island like no other 🏝️."
    },
    {
        "id": 23,
        "name": "Cultural Immersion in India",
        "weather": "tropical",
        "destination": "city",
        "activities": ["cultural", "family"],
        "budget": "economy",
        "image": "https://img.freepik.com/free-photo/man-teaching-children-about-culture-medium-shot_1258-289380.jpg?t=st=1728586649~exp=1728590249~hmac=feb8f1237c81dc1894af507d84d206ac233b8d77e476c0f407ca60598cb69f48&w=1060",
        "description": "Experience the diverse cultures and traditions of India 🕌🛕."
    },
    {
        "id": 24,
        "name": "Scandinavian Fjord Cruise",
        "weather": "temperate",
        "destination": "mountain",
        "activities": ["relaxation", "adventure"],
        "budget": "premium",
        "image": "https://img.freepik.com/free-photo/cruise-ship-sea-with-mountains_23-2148153636.jpg?t=st=1728586672~exp=1728590272~hmac=64b331cd86567a33ea6a1089b02e8d1919c7d7692d1c684e984dcf3173560c0c&w=1060",
        "description": "Sail through majestic fjords ⛰️ and enjoy stunning landscapes 🚢."
    },
    {
        "id": 25,
        "name": "Desert Safari in Dubai",
        "weather": "tropical",
        "destination": "countryside",
        "activities": ["adventure", "entertainment"],
        "budget": "luxury",
        "image": "https://img.freepik.com/free-photo/traveling-with-off-road-car_23-2151472970.jpg?t=st=1728586692~exp=1728590292~hmac=c1cb70d1f336224a7eec1d2aabd751f1647a513d0409a05bb3a68268e7761ba1&w=1060",
        "description": "Experience dune bashing 🏜️ and cultural shows 🐪 in the desert."
    }
]

class FilterItinerariesInput(BaseModel):
    weather: Optional[List[Literal["tropical", "temperate", "polar"]]] = Field(None, description="The desired weather condition(s). Choose from tropical, temperate, or polar.")
    destination: Optional[List[Literal["beach", "mountain", "city", "countryside", "island"]]] = Field(None, description="The desired destination type(s). Choose from beach, mountain, city, countryside, or island.")
    activities: Optional[List[Literal["adventure", "relaxation", "cultural", "family", "romantic", "wildlife", "entertainment", "sports", "wellness"]]] = Field(None, description="A list of desired activities. Choose from adventure, relaxation, cultural, family, romantic, wildlife, entertainment, sports, or wellness.")
    budget: Optional[List[Literal["economy", "standard", "premium", "luxury", "ultra_luxury"]]] = Field(None, description="The desired budget level(s). Choose from economy, standard, premium, luxury, or ultra_luxury.")

class FilterItinerariesTool(BaseTool):
    name = "filter_itineraries"
    description = "Use this tool to show itineraries based on weather, destination, activities, and budget filters"
    args_schema = FilterItinerariesInput

    def filter_itineraries(self, weather=None, destination=None, activities=None, budget=None):
        filtered_itins = itineraries

        def filter_condition(itin, key, value):
            if isinstance(value, list):
                return itin.get(key) in value
            return itin.get(key) == value

        for key, value in [('weather', weather), ('destination', destination), ('budget', budget)]:
            if value:
                filtered_itins = [itin for itin in filtered_itins if filter_condition(itin, key, value)]
        
        if activities:
            filtered_itins = [itin for itin in filtered_itins if any(activity in itin.get('activities', []) for activity in activities)]
        
        return filtered_itins

    def _run(self, weather: Optional[Union[str, List[str]]] = None, 
             destination: Optional[Union[str, List[str]]] = None, 
             activities: Optional[List[str]] = None, 
             budget: Optional[Union[str, List[str]]] = None) -> str:
        
        # Log or validate the input arguments
        print(f"Running with weather: {weather}, destination: {destination}, activities: {activities}, budget: {budget}")
        
        filtered_itins = self.filter_itineraries(weather, destination, activities, budget)
        
        result = "Filtered Itineraries:\n\n"
        for itin in filtered_itins:
            result += f"Name: {itin['name']}\n"
            result += f"Weather: {itin['weather']}\n"
            result += f"Destination: {itin['destination']}\n"
            result += f"Activities: {', '.join(itin['activities'])}\n"
            result += f"Budget: {itin['budget']}\n"
            result += f"Description: {itin['description']}\n\n"
        
        return result

def create_filter_itineraries_tool():
    return FilterItinerariesTool()

#------------------------------------------------------------------------------------------------

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

class PineconeSearchInput(BaseModel):
    query: str = Field(..., description="The english written query to search for in the Pinecone index")

class PineconeSearchTool(BaseTool):
    name: str = "pinecone_search"
    description: str = "Search for itineraries based on a query that summarizes the user's preferences in first person based on the conversation history"
    args_schema: type[BaseModel] = PineconeSearchInput
    vector_store: PineconeVectorStore = Field(exclude=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        index_name = "ncl-full-itineraries"
        pc = Pinecone()
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        
        if index_name in existing_indexes:
            index = pc.Index(index_name)
        else:
            raise ValueError(f"Index '{index_name}' does not exist. Please create it first.")
        
        self.vector_store = PineconeVectorStore(
            index=index,
            embedding=OpenAIEmbeddings(model="text-embedding-3-large")
        )

    def _run(self, query: str) -> list:
        results = self.vector_store.similarity_search_with_score(query, k=25)
        
        # Define a score threshold
        score_threshold = 0.500
        
        # Format results as a list of dicts with each metadata of the returned results
        formatted_results = []
        for doc, score in results:
            if score >= score_threshold:
                formatted_results.append(doc.metadata)
        
        return formatted_results

def create_pinecone_search_tool() -> PineconeSearchTool:
    return PineconeSearchTool()


#------------------------------------------------------------------------------------------------
import datetime

class DateFilterInput(BaseModel):
    start_date: str = Field(..., description="The start date in mm/dd/yyyy format")
    end_date: str = Field(..., description="The end date in mm/dd/yyyy format")

class DateFilterTool(BaseTool):
    name: str = "date_filter"
    description: str = "Filter itineraries based on start and end dates after having performed a search"
    args_schema: type[BaseModel] = DateFilterInput

    def _run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        # start_date_obj = datetime.datetime.strptime(start_date, "%m/%d/%Y").date()
        # end_date_obj = datetime.datetime.strptime(end_date, "%m/%d/%Y").date()
        
        return {
            "start_date": start_date,
            "end_date": end_date,
        }

def create_date_filter_tool() -> DateFilterTool:
    return DateFilterTool()





