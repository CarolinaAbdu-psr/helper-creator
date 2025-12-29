import argparse
import yaml
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
# single-agent config file at repository root
AGENT_CONFIG_PATH = "agent.yaml"


# -----------------------------
# Load agent configuration
# -----------------------------

def load_agent_config(filepath: str) -> bool:
    """Load prompts and templates from the agent YAML configuration.

    Expected structure (agent.yaml):
    prompts:
      system_prompt_template: |
        ...
      prepare_prompt: |
        ...
      generation_task_prompt:
        user_prompt_sddp_generation: |
          ...
    """
    global SYSTEM_PROMPT_TEMPLATE, GENERATION_TASK_PROMPT, PREPARE_PROMPT
    SYSTEM_PROMPT_TEMPLATE = None
    GENERATION_TASK_PROMPT = None
    PREPARE_PROMPT = None
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        prompts = config.get('prompts', {}) if isinstance(config, dict) else {}
        # system prompt
        SYSTEM_PROMPT_TEMPLATE = prompts.get('system_prompt_template') or prompts.get('system_prompt')
        # prepare prompt (for the prepare step)
        PREPARE_PROMPT = prompts.get('prepare_prompt')
        # generation task prompt (nested mapping, for the execution step)
        gen = prompts.get('generation_task_prompt') or {}
        if isinstance(gen, dict):
            GENERATION_TASK_PROMPT = gen.get('user_prompt_sddp_generation')
        elif isinstance(gen, str):
            GENERATION_TASK_PROMPT = gen

        if not SYSTEM_PROMPT_TEMPLATE:
            raise KeyError('system_prompt_template not found in agent config')

        logger.info(f"Agent template loaded successfully from {filepath}")
        return True
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logger.error(f"Error loading Agent from {filepath}: {e}")
        return False


def load_agents_config() -> Dict[str, Any]:
    """Load agent configuration file from AGENT_CONFIG_PATH."""
    filepath = AGENT_CONFIG_PATH
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
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}

        workflow = StateGraph(AgentState)
        # prepare: build a natural-language reference script (properties + examples)
        workflow.add_node("prepare", self.prepare_reference)
        workflow.add_node("llm", self.call_llm)
        workflow.add_node("retriver", self.take_action)

        # flow: prepare -> llm -> (if tool calls) retriver -> llm -> ...
        workflow.add_edge('prepare', 'llm')
        workflow.add_conditional_edges(
            'llm',
            self.exists_action,
            {True: 'retriver', False: END}
        )
        workflow.add_edge('retriver', 'llm')
        workflow.set_entry_point('prepare')
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

    def prepare_reference(self, state: AgentState):
        """
        Build a natural-language reference script the agent will use while invoking tools.
        Uses external retriever functions (retrive_properties, retrive_examples)
        to fetch RAG context, then invokes the LLM to generate a reference script
        using PREPARE_PROMPT from agent.yaml.
        
        Returns: Both the reference script AND a follow-up prompt instructing the LLM
        to execute the plan using tools (from GENERATION_TASK_PROMPT).
        """
        try:
            # last user input
            last_user_input = state['messages'][-1].content if state.get('messages') else ''

            # Retrieve properties and examples using external retrievers
            try:
                vectorstore = load_vectorstore("properties")
                retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
                last_message_content = state["messages"][-1].content
                docs = retriever.invoke(last_message_content)
                properties_str = format_properties(docs)
            except Exception as e:
                logger.warning(f"Failed to retrieve properties: {e}")
                properties_str = "(unable to load properties)"

            try:
                examples_dict = retrive_examples(state)
                examples_str = examples_dict.get('examples', '(no examples found)')
            except Exception as e:
                logger.warning(f"Failed to retrieve examples: {e}")
                examples_str = "(unable to load examples)"

            prompt = PREPARE_PROMPT

            system_message = SystemMessage(content=self.system)
            user_prompt_content = prompt.format(
                input=last_user_input,
                properties=properties_str,
                examples = examples_str)

            human_message = HumanMessage(content=user_prompt_content)

            messages = [system_message,human_message] 
            reference_script = self.model.invoke(messages)

            logger.info(f"Script Generated : \n {reference_script.content}")
            
            # Now prepare the execution phase: add a follow-up message with GENERATION_TASK_PROMPT
            # This instructs the LLM to use the tools to implement the reference script
            execution_prompt = GENERATION_TASK_PROMPT
            execution_message = HumanMessage(content=execution_prompt)
            return {'messages': [reference_script, execution_message]}
        except Exception as e:
            logger.error(f"prepare_reference failed: {e}")
            # fallback to empty message so workflow proceeds
            return {'messages': [HumanMessage(content='(prepare_reference failed)')]} 

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
def get_vectorstore_directory(doc_type: str) -> str:
    return f'vectorstores/{doc_type}'

def load_vectorstore(doc_type: str) -> Chroma:
    """1.1. Loads the vectorstore with examples to use as context """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    persist_directory = get_vectorstore_directory(doc_type)

    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
    raise ValueError(f"Vectorstore not found : {persist_directory}")

# -----------------------------
# Tools
# -----------------------------

def format_contrastive_examples(docs: List) -> str:
    """
    Formats a list of retrieved documents into a context string
    that includes both Correct and Incorrect Codes.
    """
    formatted_blocks = []
    
    for i, doc in enumerate(docs):
        # 1. Get metadata
        metadata = doc.metadata
        
        
        # 3. Create example
        block = f"""
        ### EXample {i + 1}
        Question: {doc.page_content}

        CORRECT SINTAX (DO): ({metadata.get('correct_inst', 'N/A')})
        ```python
        {metadata.get('correct_code', 'N/A')}````

        INCORRECT SINTAX (DON'T DO): ({metadata.get('incorrect_code', 'N/A')})"""

        formatted_blocks.append(block.strip())
        
    return "\n\n" + "\n\n".join(formatted_blocks)

def retrive_examples(state:AgentState)->Dict[str,Any]:
    """
    1.2 Get Code examples
    """
    try:
        doc_type = "examples"
        vectorstore = load_vectorstore(doc_type)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        # prefer explicit input, fallback to last message content
        query = state.get("input") if isinstance(state, dict) and state.get("input") else state.get("messages", [])[-1].content
        docs = retriever.invoke(query)
        examples_str = format_contrastive_examples(docs)
        print(examples_str)
        return {"examples": examples_str}
    except Exception as e:
        logger.exception("retrive_examples failed")
        return {"examples": f"TOOL_ERROR: retrive_examples failed: {type(e).__name__}: {e}"}

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

@tool
def retrive_properties(state:AgentState)->str:
    """
    Retrieve detailed information about available object types and their properties from the SDDP study.

    Tool behavior:
    - Input: `state` (AgentState) where `state['messages'][-1]` contains the user's query or context.
    - Output: a single string containing a concise, machine-readable description of:
      * mandatory properties for each relevant object type
      * reference properties (names and expected target types)
      * static and dynamic properties and how to set them

    Usage guidance for the LLM (important):
    - Call this tool BEFORE attempting to create objects.
    - Use the property names and reference keys returned here EXACTLY when invoking
      `create_objects` or `set_static_properties`.
    - The returned text is formatted for programmatic consumption; prefer extracting
      exact property names and examples rather than paraphrasing.

    Returns: Formatted documentation of available objects and their properties.
    """
    try:
        vectorstore = load_vectorstore("properties")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        last_message_content = state["messages"][-1].content
        docs = retriever.invoke(last_message_content)
        properties_str = format_properties(docs)
        return properties_str
    except Exception as e:
        logger.exception("retrive_properties failed")
        return f"TOOL_ERROR: retrive_properties failed: {type(e).__name__}: {e}"


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

@tool 
def get_correspondent_objct_type(reftype:str):
    """
    Maps an SDDP reference property name to its corresponding valid object type(s).
    Use this tool whenever you encounter a property starting with 'Ref' and need to 
    identify which object type must be created or searched for.

    Args:
        reftype (str): The name of the reference property found in the schema or script 
                       (e.g., 'RefPlants', 'RefBus', 'RefCircuits').

    Returns:
        str | list: A single object type string or a list of possible valid object types.
                    If a list is returned, you must determine the correct specific type 
                    based on the context of the user's request.

    Example:
        - If 'RefPlants' is passed, it returns ['ThermalPlant', 'HydroPlant', 'RenewablePlant'].
        - If 'RefBus' is passed, it returns 'Bus'.
    """
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

def check_mandatory_refs(refs:Dict):
    # If there are no refs or empty dict, nothing to check
    logger.debug(f"Checking mandatory references: {refs}")
    if not refs:
        return True
    if refs: 
        for ref_type, items in refs.items():
            if not isinstance(items,list):
                logger.info("TOOL_ERROR: 'refs' values must be lists of tuples (ObjType, code).")
                return "TOOL_ERROR: 'refs' values must be lists of tuples (ObjType, code)."
            for item in items: 
                if not(isinstance(item,tuple) or (isinstance(item,list) and len(item)==2)):
                    print(item,not isinstance(item,tuple), not (isinstance(item,list) and len(item)==2) )
                    print("TOOL_ERROR: Each item in the list must be a (Type, Code) pair.")
                    return "TOOL_ERROR: Each item in the list must be a (Type, Code) pair."
                
                if len(STUDY.find_by_code(item[0], item[1])) == 0 :
                    return f"Referenced object not found: type_candidates={item[0]}, code={item[1]}"
    return True

@tool 
def required_references(object_type:str)->str:
    """
    Retrieve the mandatory refernces for a given object type.

    Tool behavior:
    - Input: `object_type` (str): the exact object type name.
    - Output: A comma-separated string listing all mandatory references names for the specified object type.

    Usage guidance for the LLM:
    - Call this tool BEFORE attempting to create objects of the specified type.
    - Use the exact property names returned here when invoking `create_objects`.
    """
    try:
        obj = STUDY.create(object_type)
        mandatory_props = []
        desc = obj.descriptions()
        for item in desc:
            if obj.description(item).is_required() and obj.description(item).is_reference():
                mandatory_props.append(item)
        return ", ".join(mandatory_props)
    except Exception as e:
        logger.exception("required_properties failed")
        return f"TOOL_ERROR: required_properties failed: {type(e).__name__}: {e}"
    

@tool 
def get_available_object_types() -> str:
    """
    Retrieve a list of all available object types in the SDDP study.

    Tool behavior:
    - No inputs.
    - Output: A comma-separated string listing all object type names.

    Usage guidance for the LLM:
    - Call this tool to discover which object types can be created or manipulated.
    - Use the exact object type names returned here when invoking other tools.
    """

    try:
        objects = [
    "ACInterconnection",
    "Area",
    "Battery",
    "Bus",
    "BusShunt",
    "CSP",
    "CircuitFlowConstraint",
    "DCBus",
    "DCLine",
    "Demand",
    "DemandSegment",
    "Emission",
    "EnergyEfficiency",
    "ExpansionAssociated",
    "ExpansionCapacity",
    "ExpansionDecision",
    "ExpansionExclusive",
    "ExpansionGenericConstraint",
    "ExpansionPrecedence",
    "ExpansionProject",
    "ExpansionSatisfaction",
    "FlowController",
    "Fuel",
    "FuelConsumption",
    "FuelContract",
    "FuelProducer",
    "FuelReservoir",
    "GasNode",
    "GasNonThermalDemand",
    "GasPipeline",
    "GenerationConstraint",
    "GenericConstraint",
    "GenericVariable",
    "HydroGenerator",
    "HydroPlant",
    "HydroPlantConnection",
    "HydroStation",
    "HydroStationConnection",
    "Interconnection",
    "InterpolationGenericConstraint",
    "LCCConverter",
    "LineReactor",
    "Load",
    "MTDCLink",
    "Owner",
    "Ownership",
    "PSRElectrificationProducer",
    "PaymentSchedule",
    "PowerInjection",
    "RenewableCapacityProfile",
    "RenewableGenerator",
    "RenewablePlant",
    "RenewableStation",
    "RenewableTurbine",
    "RenewableWindSpeedPoint",
    "ReserveGeneration",
    "ReservoirSet",
    "SensitivityGroup",
    "SeriesCapacitor",
    "StaticVarCompensator",
    "SumOfCircuits",
    "SumOfConstraints",
    "SumOfInterconnections",
    "SupplyChainDemand",
    "SupplyChainDemandSegment",
    "SupplyChainFixedConverter",
    "SupplyChainFixedConverterCommodity",
    "SupplyChainNode",
    "SupplyChainProcess",
    "SupplyChainProducer",
    "SupplyChainStorage",
    "SupplyChainTransport",
    "SynchronousCompensator",
    "System",
    "TargetGeneration",
    "ThermalCombinedCycle",
    "ThermalGenerator",
    "ThermalPlant",
    "ThreeWindingsTransformer",
    "Transformer",
    "TransmissionLine",
    "VSCConverter",
    "Waterway",
    "Zone"
        ]
        return ", ".join(objects)
    except Exception as e:
        logger.exception("get_available_object_types failed")
        return f"TOOL_ERROR: get_available_object_types failed: {type(e).__name__}: {e}"

@tool 
def check_object_properties(objct_type:str):
    """Returns the available properties for a given object type"""
    try:
        obj = STUDY.create(objct_type)
        return obj.help()
    except Exception as e:
        logger.exception("check_object_properties failed")
        return f"TOOL_ERROR: check_object_properties failed: {type(e).__name__}: {e}"


def set_references(obj, ref_name:str, ref_obj_type:str,  ref_code:int):
    """
    Establishes a logical link between two SDDP objects. 
    Crucial for defining dependencies like linking a ThermalPlant to its Fuel.

    Args:
        obj_type (str): The type of the parent object receiving the reference (e.g., 'ThermalPlant').
        obj_code (int): The unique code of the parent object.
        ref_name (str): The property name of the reference (e.g., 'RefFuel' or 'RefFuels').
        ref_obj_type (str): The type of the object being referenced. 
                           Use 'get_correspondent_objct_type' to find the correct type.
        ref_code (int): The unique code of the object to be referenced.

    Returns:
        str: A success message or an error description if the link fails.
    """
    
    try:
        atual_refs = obj.get(ref_name)

        ref_obj = STUDY.find_by_code(ref_obj_type,ref_code)[0]

        if atual_refs is None: 
            if ref_obj_type.endswith("s") and not ref_obj_type=="RefBus": 
                atual_refs = []
            else :
                atual_refs=ref_obj

        if isinstance(atual_refs,list):
            atual_refs.append(ref_obj)
            obj.set(ref_name,atual_refs)
        
        else: 
            obj.set(ref_name, atual_refs)

    except Exception as e:
        logger.warning(f"Failed to set reference {ref_name}({ref_obj_type}) on {obj.type} (code={obj.code}): {e}")
        return f"Failed to set reference {ref_name}({ref_obj_type}) on {obj.type} (code={obj.code}): {e}"
    

@tool 
def create_objects(object_type:str, name:str, id:str, code:int, refs:Dict=None): 
    """
    Initializes and adds a new object to the SDDP STUDY. This is the primary creation tool.

    Args:
        object_type (str): The exact class name (e.g., 'ThermalPlant').
        name (str): The full name of the object.
        id (str): A 2-character short identifier.
        code (int): A unique numeric integer code.
        refs (Dict[str, List[Tuple[str, int]]], optional): A dictionary where:
            - Key: The reference property name (e.g., "RefFuels", "RefBus").
            - Value: A LIST of Tuples, where each tuple is (TargetObjectType, TargetCode).
            Example: { "RefFuels": [("Fuel", 1)], "RefBuses": [("Bus", 10), ("Bus", 11)] }
            Note: Even for a single reference, the value MUST be a list containing one tuple.

    Returns:
        str: Success message or a TOOL_ERROR.
    """
    try:
        # Validate basic properties first
        basic_check = check_basic_properties(object_type, code, name, id)
        if basic_check is not True:
            return f"Basic property validation failed: {basic_check}"

        # Validate referenced objects
        refs_check = check_mandatory_refs(refs)
        if refs_check is not True:
            return f"Reference validation failed: {refs_check}"

        # Create the object
        obj = STUDY.create(object_type)
        obj.name = name
        obj.code = code
        obj.id = id

        if refs: 
            logger.info(f"Setting References for {refs}")
            for ref_type, items in refs.items():
                logger.info(ref_type,items)
                if not isinstance(items,list):
                    logger.info("TOOL_ERROR: 'refs' values must be lists of tuples (ObjType, code).")
                    return "TOOL_ERROR: 'refs' values must be lists of tuples (ObjType, code)."
                for item in items: 
                    print(obj)
                    set_references(obj,ref_type,item[0],item[1])
                    

        STUDY.add(obj)

        logger.info(f"Object {obj} created with success")
        return f"Object {obj} created with success"
                

    except Exception as e:
        logger.exception(f"TOOL_ERROR: create_objects failed: {type(e).__name__}: {e}")
        return f"TOOL_ERROR: create_objects failed: {type(e).__name__}: {e}"
    

def check_object_exist(objtyppe,code):
    obj = STUDY.find_by_code(objtyppe,code)
    if len(obj)==0 :
        return "Object of type and code doens't exists"
    
    return True


@tool 
def set_static_properties(objtype, code, properties:Dict[str,Any]):
    """
    Set one or more static properties on an existing object.

    Tool behavior:
    - Inputs:
      * `objtype` (str): exact object type name
      * `code` (int): object code to find the existing object
      * `properties` (Dict[str, Any]): mapping property_name -> value
    - Validates object existence and that each property is static (not a dataframe).
    - Returns a success message or a clear error string.

    Usage guidance for the LLM:
    - Use property names exactly as returned by `retrive_properties`.
    - If a property is dynamic (time-series), call `set_dynamic_properties` instead.
    """
    try:
        exist = check_object_exist(objtype,code)
        if exist is not True:
            return f"Object existence check failed: {exist}"
        obj = STUDY.find_by_code(objtype,code)[0]
        for prop, value in properties.items():
            description = obj.description(prop)
            is_dataframe = description.is_dynamic() and len(description.dimensions()) > 0
            if is_dataframe:
                return "Can't create dataframe with this function. Please use set_dynamic_properties"
            obj.set(prop, value)
        return f"Set static properties on {objtype} code={code}"
    except Exception as e:
        logger.exception("set_static_properties failed")
        return f"TOOL_ERROR: set_static_properties failed: {type(e).__name__}: {e}"

@tool
def save_study():
    """
    Persist the current `STUDY` to disk.

    Tool behavior:
    - No inputs.
    - Side-effect: saves study to `./study_llm_test` (currently hard-coded).
    - Output: success string or exception message.

    Usage guidance for the LLM:
    - Call this tool when all objects/properties have been created and you want to
      persist the study.
    """
    try:
        study_path = "./study_llm_test"
        psr.factory.save_study(study_path)
        logger.info("Study saved with success")
        return "Study saved with success"
    except Exception as e:
        logger.exception("save_study failed")
        return f"TOOL_ERROR: save_study failed: {type(e).__name__}: {e}"



if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="RAG Agent - Translate questions to Cypher and answer them with RAG + auto-fix tools")
    parser.add_argument("-m", "--model", default="gpt-4.1", help="LLM model to be used. Default: gpt-4.1")
    #parser.add_argument("-s", "--study_path", required=True, help="Path to the study files required to load the graph schema into Neo4j.")
    parser.add_argument("-q", "--query", required=True, help="Natural language question to be processed by the agent.")
    args = parser.parse_args()
    model = args.model
    #study_path = args.study_path
    user_input = args.query


    logger.info("--- Starting RAG Agent ---")
    logger.info(f"Selected model: {model}")
    #logger.info(f"Study path: {study_path}")
    logger.info(f"User question: {user_input}")
    try:
        global STUDY
        STUDY = psr.factory.create_study()
        system = STUDY.find("System")[0]
        STUDY.remove(system)
        logger.info("Study loaded successfully.")
        
        llm = initialize(model)
        logger.info("LLM initialized.")
        
        # Register available tools for the agent. Include all tool functions the
        # LLM might call during the workflow.
        tools = [
            check_object_properties,
            required_references,
            get_correspondent_objct_type,
            get_available_object_types,
            create_objects,
            set_static_properties,
            save_study,
        ]

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

        study_path = "./study_llm_test"
        psr.factory.save_study(study_path)
        logger.info(f"Study saved successfully to {study_path}.")

    except Exception as e:
        logger.error("--- CRITICAL ERROR ---")
        logger.error(f"Error Detail: {type(e).__name__}: {str(e)}")
        sys.exit(1)
