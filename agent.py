import requests
from typing import Sequence, Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

# FHIR API

BASE_URL = "https://hapi.fhir.org/baseR4"


# LOADING Groq LLM

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# STATE

# AgentState: its purpose is to tell the langGraph what to keep track while the workflow is running
class AgentState(TypedDict):
    """The state of the Agent"""
    messages: Annotated[
        Sequence[BaseMessage],
        add_messages
    ]
    on_topic: bool



# QUESTION CLASSIFICATION: To check for related question...............


class QuestionClassification(BaseModel):

    is_hospital_related: bool = Field(
        description=(
            "True if the question is related to doctors, hospitals, "
            "patients, appointments, or appointment availability."
        )
    )


classifier_llm = llm.with_structured_output(
    QuestionClassification,
    method="json_mode"
)


def question_classifier(state: AgentState):

    print("Entering question_classifier")

    question = state["messages"][-1].content

    result = classifier_llm.invoke(
            f"""
    Determine whether this question is related to a hospital
    appointment assistant.

    The assistant can help with:

    - finding doctors
    - doctor specialties
    - checking appointment availability
    - booking appointments
    - hospitals
    - patients
    - hospital appointments

    Question:
    {question}

    Respond with JSON only.

    The JSON must have this exact structure:

    {{
        "is_hospital_related": true
    }}

    Return true if the question is related to the hospital
    appointment assistant.

    Return false if the question is not related.
    """
    )

    print(
        f"Hospital related: "
        f"{result.is_hospital_related}"
    )

    return {
        "on_topic": result.is_hospital_related
    }


# SEARCH DOCTORS


@tool
def search_doctors(specialty: str):
    """Search for doctors based on their medical specialty."""

    response = requests.get(
        f"{BASE_URL}/PractitionerRole",
        params={
            "specialty:text": specialty
        }
    )

    response.raise_for_status()

    data = response.json()

    return data


# CHECK AVAILABILITY

@tool
def check_availability(doctor: str, date: str):
    """Check available appointment slots for a doctor on a specific date."""

    response = requests.get(
        f"{BASE_URL}/Slot",
        params={
            "status": "free"
        }
    )

    response.raise_for_status()

    data = response.json()

    return data


# BOOK APPOINTMENT


@tool
def book_appointment(
    doctor: str,
    date: str,
    time: str
):
    """Book an appointment with a doctor."""

    appointment_data = {
        "resourceType": "Appointment",
        "status": "proposed",
        "description": f"Appointment with {doctor}",
        "start": f"{date}T{time}"
    }

    response = requests.post(
        f"{BASE_URL}/Appointment",
        json=appointment_data,
        headers={
            "Content-Type": "application/fhir+json"
        }
    )

    response.raise_for_status()

    return response.json()



# ALL TOOLS


tools = [
    search_doctors,
    check_availability,
    book_appointment
]


# Give Gemini access to the tools

llm_with_tools = llm.bind_tools(tools)


# AGENT

def agent(state: AgentState):

    print("Entering agent")

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# TOPIC ROUTER

def topic_router(state: AgentState):

    print("Entering topic_router")

    if state["on_topic"]:

        print("Question is hospital-related")

        return "agent"

    print("Question is off-topic")

    return "off_topic"


# CHECK WHETHER AGENT WANTS TO USE A TOOL

def should_continue(state: AgentState):

    print("Entering should_continue")

    last_message = state["messages"][-1]

    if last_message.tool_calls:

        print("Agent wants to use a tool")

        return "tools"

    print("Agent does not need a tool")

    return "end"


# OFF-TOPIC RESPONSE

def off_topic_response(state: AgentState):

    print("Entering off_topic_response")

    return {
        "messages": [
            AIMessage(
                content=(
                    "I'm sorry! I can only help with "
                    "doctors, hospitals, patients, "
                    "availability and appointments."
                )
            )
        ]
    }


# CANNOT ANSWER

def cannot_answer(state: AgentState):

    print("Entering cannot_answer")

    return {
        "messages": [
            AIMessage(
                content=(
                    "I'm sorry, but I cannot find "
                    "the information you're looking for."
                )
            )
        ]
    }


# TOOL NODE

tool_node = ToolNode(tools)

#saving....
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

builder = StateGraph(AgentState)


# Add nodes

builder.add_node("classifier", question_classifier)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)
builder.add_node("off_topic", off_topic_response)
builder.add_node("cannot_answer", cannot_answer)


# EDGES

# START → CLASSIFIER

builder.add_edge(START, "classifier")

# CLASSIFIER → AGENT or OFF_TOPIC

builder.add_conditional_edges("classifier",
    topic_router,
    {
        "agent": "agent",
        "off_topic": "off_topic"
    }
)

# AGENT → TOOLS or END

builder.add_conditional_edges("agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)


# TOOLS → AGENT

builder.add_edge("tools","agent")


# OFF_TOPIC → END

builder.add_edge("off_topic",END)


# COMPILE GRAPH

graph = builder.compile()