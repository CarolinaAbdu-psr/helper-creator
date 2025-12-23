import json
import argparse
import yaml
import datetime as dt
import os
import operator
from typing import Tuple, List, Annotated, Dict, Any
import psr.factory

import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, AnyMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.tools import tool
from langchain_chroma import Chroma
import chromadb.config
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# Load environment variables from .env file
load_dotenv()

REQUEST_TIMEOUT = 120
MAX_TOKENS = 4096

# Agent template settings
AGENTS_DIR = "agent"
AGENT_FILENAME = "agent.yaml"


# -----------------------------
# Load agent configuration
# -----------------------------

def load_agent_config(filepath: str) -> bool:
    """Load prompts and templates from the agent YAML configuration."""
    global SYSTEM_PROMPT_TEMPLATE
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        SYSTEM_PROMPT_TEMPLATE = config['system_prompt_template']

        logger.info(f"Agent template loaded successfully from {filepath}")
        return True
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logger.error(f"Error loading Agent from {filepath}: {e}")
        return False


def load_agents_config() -> Dict[str, Any]:
    """Load agent configuration file."""
    filepath = os.path.join(AGENTS_DIR, AGENT_FILENAME)
    if load_agent_config(filepath):
        return {'status': 'loaded'}
    return {'status': 'failed'}


# Load configuration once
load_agents_config()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage],operator.add] # Add messages to state (humam or ai result)

class RAGAgent:

    def __init__(self, model, tools, system):
        self.system = system
        # Bind tools to the model so it knows it can call them
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}

        workflow = StateGraph(AgentState)
        workflow.add_node()
        workflow.add_node("llm", self.call_llm)
        workflow.add_node("retriver",self.take_action)

        workflow.add_conditional_edges(
            'llm',
            self.exists_action,
            {True:'retriver',False: END}
        )
        workflow.add_edge('retriver','llm')
        workflow.set_entry_point('llm')
        self.workflow = workflow.compile()

    def exists_action(self,state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls)>0 
    
    def call_llm(self, state: AgentState):
        messages = state['messages'] 
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)  # AI response
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls  # use tools 
        results = []
        for t in tool_calls:
            logger.info(f"Calling Tool: {t}")
            if not t['name'] in self.tools:
                logger.warning(f"Tool {t} doesn't exist")
                result = "Incorrect tool name. Retry and select an available tool"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        logger.info("Tools Execution Complete. Back to the model")
        return {'messages': results}
    


def initialize(model: str) :
    """Initialize the LLM and return the compiled LangGraph workflow and memory."""

    try:
        if model == "gpt-5-2025-08-07":
            llm = ChatOpenAI(model_name="gpt-5-2025-08-07", request_timeout=REQUEST_TIMEOUT)
        elif model == "gpt-4.1":
            llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)
        elif model == "gpt-4.1-mini":
            llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)
        elif model == "o3":
            llm = ChatOpenAI(model_name="o3", request_timeout=REQUEST_TIMEOUT)
        elif model == "claude-4-sonnet":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model='claude-sonnet-4-20250514', anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'), temperature=0.7, max_tokens=MAX_TOKENS, timeout=REQUEST_TIMEOUT)
        elif model == "deepseek-reasoner":
            llm = BaseChatOpenAI(model='deepseek-reasoner', openai_api_key=os.getenv('DEEPSEEK_API_KEY'), openai_api_base='https://api.deepseek.com', temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)
        elif model == "local_land":
            llm = ChatOpenAI(model_name="qwen3:14b", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT, base_url= "http://10.246.47.184:10000/v1")
        else:
            llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)

        return llm
    
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")
        raise

# -----------------------------
# Retrive context
# -----------------------------

def load_vectorstore() -> Chroma:
    """Load a Chroma vectorstore persisted in `vectorstore` directory."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = "vectorstore"
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
    raise ValueError(f"Vectorstore not found: {persist_directory}")



def format_properties(docs: List) -> str:
    """
    Formats a list of retrieved documents into a context string
    that includes availables properties
    """
    formatted_blocks = []
    
    available_objects = "The available objects to be used at study.find() or study.create() are the following: ACInterconnection, Area, Battery, Bus, BusShunt, Circuit, CircuitFlowConstraint, CSP, DCBus, DCLine, Demand, DemandSegment, Emission, FlowController, Fuel, FuelConsumption, FuelContract, FuelProducer, FuelReservoir, GasEmission, GasNode, GasPipeline, GenerationConstraint, GenericConstraint, HydroGenerator, HydroPlant, HydroPlantConnection, HydroStation, HydroStationConnection, Interconnection, InterpolationGenericConstraint, LCCConverter, LineReactor, Load, MTDCLink, PaymentSchedule, PowerInjection, RenewableCapacityProfile, RenewableGenerator, RenewablePlant, RenewableStation, RenewableTurbine, RenewableWindSpeedPoint, ReserveGeneration, ReservoirSet, SensitivityGroup, SeriesCapacitor, StaticVarCompensator, SumOfCircuits, SumOfInterconnections, SupplyChainDemand, SupplyChainDemandSegment, SupplyChainFixedConverter, SupplyChainFixedConverterCommodity, SupplyChainNode, SupplyChainProcess, SupplyChainProducer, SupplyChainStorage, SupplyChainTransport, SynchronousCompensator, System, TargetGeneration, ThermalCombinedCycle, ThermalGenerator, ThermalPlant, ThreeWindingsTransformer, Transformer, TransmissionLine, TwoTerminalDCLink, VSCConverter, Waterway, Zone"
    
    formatted_blocks.append(available_objects)

    for i, doc in enumerate(docs):

        # 1. Get object name and metadata (properties)
        metadata = doc.metadata
        objct_name = doc.page_content
        
        # 3. Create example
        block = f"""
        Object Name: {objct_name}

        Madatory properties to create {objct_name}: {metadata.get("mandatory")}

        Reference properties wich must be used to link objects: {metadata.get("references_objects")}

        Static properties which can be acessed with .get(PropertyName) function and created by .set(PropertyName,value) function : {metadata.get("static_properties")}

        Dynamic properties which can be acessed with .get_df(PropertyName) or .get_at(PropertyName, date) functions and created by .set_df(df) 
        or .set_at(PropertyName, date, value) function : {metadata.get("dynamic_properties")}
        """

        formatted_blocks.append(block.strip())
        
    return "\n\n" + "\n\n".join(formatted_blocks)

# -----------------------------
# Tools
# -----------------------------

@tool
def retrive_properties(state:AgentState)->str:
    """
    Retrieve detailed information about available object types and their properties from the SDDP study.
    
    Use this tool FIRST to understand:
    - What object types exist (e.g., ThermalPlant, HydroPlant, Bus)
    - What mandatory properties are needed to create each object
    - What static properties can be accessed with tool get_static_property 
    - What dynamic properties can be accessed 
    - What reference properties link objects together
    
    Returns: Formatted documentation of available objects and their properties.
    Use the property names returned here when calling other tools.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    last_message_content = state["messages"][-1].content
    docs = retriever.invoke(last_message_content)
    properties_str = format_properties(docs)
    return properties_str


def check_basic_properties(objtype, code, name, id):
    # Code check
    if not isinstance(code, int) or code > 99:
        return "Code must be an integer lower than 99"
    if len(STUDY.find_by_code(objtype, code)) > 0:
        return f"Code {code} already registered for {objtype}. Choose a new code"
    # Name check
    if not isinstance(name, str) or len(name) > 12:
        return "Name must be a string with 12 characters maximum"
    # ID check
    if not isinstance(id, str) or len(id) > 2:
        return "Id must be a string with 2 characters maximum"
    if len(STUDY.find_by_id(objtype, id)) > 0:
        return f"ID {id} is already been used for {objtype}. Choose a different one"

    return True

def get_correspondent_objct_type(reftype:str):
    mapping = {
        "RefDemand": "Demand",
        "RefBus": "Bus",
        "RefSegment": "DemandSegment",
        "RefGenerators": ["ThermalGenerator","HydroGenerator","RenewableGenerator"],
        "RefCircuit": "TransmissionLine",
        "RefArea": "Area",
        "RefZone": "Zone",
        "RefSystem": "System",
        "RefPlants": ["ThermalPlant", "HydroPlant", "RenewablePlant"],
        "RefFuels":"Fuel",
        "RefGasNode":"GasNode",
        "RefNodes":"GasNode",
        "RefStation": ["RenewableStation","HydroStation"],
        "RefStations": "HydroStation",
        "RefInterconnections": "Interconnection",
        "RefControlledBus": "Bus",
        "RefCircuits": ["TransmissionLine","Transformer","ACInterconnection","SeriesCapacitor"],
        "RefBuses": ["DCBus","Bus"],
        "RefLink": "MTDCLink",
        "RefReservoirs": "FuelReservoir",
        "RefFuelReservoirs": "FuelReservoir",
        "RefFuelProducers": "FuelProducer",
        "RefSegment": "SupplyChainDemandSegment",
        "RefProcess": "SupplyChainProcess",
    }
    # Return mapped object type; fall back to removing leading 'Ref' if unknown
    if reftype in mapping:
        return mapping[reftype]
    if isinstance(reftype, str) and reftype.startswith("Ref"):
        return reftype[3:]
    return reftype

def check_mandatory_refs(refs:Dict[str,list]):
    # If there are no refs or empty dict, nothing to check
    if not refs:
        return True
    for reftype, codes in refs.items():
        objtypes = get_correspondent_objct_type(reftype)
        # normalize to list of candidate types
        candidates = objtypes if isinstance(objtypes, (list, tuple)) else [objtypes]
        # allow single int or list of codes
        if codes is None:
            return f"Reference {reftype} provided without codes"
        codes = codes if isinstance(codes, (list, tuple)) else [codes]
        for code in codes:
            # check if any candidate type has the object with given code
            found = False
            for candidate in candidates:
                objs = STUDY.find_by_code(candidate, code)
                if len(objs) > 0:
                    found = True
                    break
            if not found:
                return f"Referenced object not found: type_candidates={candidates}, code={code}"
    return True

@tool 
def create_objects(object_type:str, name:str, id:str, code:int, refs:Dict[str,list]): 
    # Validate basic properties first
    basic_check = check_basic_properties(object_type, code, name, id)
    if basic_check is not True:
        return f"Basic property validation failed: {basic_check}"

    # Validate referenced objects
    refs_check = check_mandatory_refs(refs)
    if refs_check is not True:
        return f"Reference validation failed: {refs_check}"

    # Create the object
    try:
        obj = STUDY.create(object_type)
        obj.name = name
        obj.code = code
        obj.id = id
    except Exception as e:
        return f"Failed to create object {object_type}: {e}"

    # Set references if any
    if refs:
        for reftype, codes in refs.items():
            try:
                obj.set(reftype, codes)
            except Exception as e:
                return f"Failed to set reference {reftype} on {object_type} (code={code}): {e}"

    return f"Created {object_type} with code={code} id={id}"
    
def check_object_exist(objtyppe,code):
    obj = STUDY.find_by_code(objtyppe,code)
    if len(obj)==0 :
        return "Object of type and code doens't exists"
    
    return True


@tool 
def set_static_properties(objtype, code, properties:Dict[str,Any]):
    check_object_exist(objtype,code)
    obj = STUDY.find_by_code(objtype,code)[0]
    for prop, value in properties.items():
        description = obj.description(prop)
        is_dataframe = description.is_dynamic() and len(description.dimensions()) > 0
        if is_dataframe:
            return "Can't create dataframe with this function. Please use set_dynamic_properties"
        obj.set(prop, value)
    pass



if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="RAG Agent - Translate questions to Cypher and answer them with RAG + auto-fix tools")
    parser.add_argument("-m", "--model", default="gpt-4.1", help="LLM model to be used. Default: gpt-4.1")
    parser.add_argument("-s", "--study_path", required=True, help="Path to the study files required to load the graph schema into Neo4j.")
    parser.add_argument("-q", "--query", required=True, help="Natural language question to be processed by the agent.")
    args = parser.parse_args()
    model = args.model
    study_path = args.study_path
    user_input = args.query


    logger.info("--- Starting RAG Agent ---")
    logger.info(f"Selected model: {model}")
    logger.info(f"Study path: {study_path}")
    logger.info(f"User question: {user_input}")
    try:
        global STUDY
        STUDY = psr.factory.create_study()
        logger.info("Study loaded successfully.")
        
        llm = initialize(model)
        logger.info("LLM initialized.")
        
        tools = [retrive_properties]
        
        # Initialize agent with system prompt and user query
        initial_message = HumanMessage(content=user_input)
        messages = [initial_message]
        
        # Create agent with system prompt (as string, not list)
        agent = RAGAgent(llm, tools, SYSTEM_PROMPT_TEMPLATE)
        logger.info("Agent initialized with tools and system prompt.")
        
        # Invoke workflow
        result = agent.workflow.invoke({'messages': messages})
        logger.info("Workflow execution completed.")
        
        # Extract and display final response
        if "messages" in result and result["messages"]:
            final_response = result["messages"][-1].content
            print("\n==============================================")
            print("AGENT FINAL RESPONSE:")
            print(final_response)
            print("==============================================\n")
        else:
            logger.warning("Workflow executed but no answer generated.")
    except Exception as e:
        logger.error("--- CRITICAL ERROR ---")
        logger.error(f"Error Detail: {type(e).__name__}: {str(e)}")
        sys.exit(1)
