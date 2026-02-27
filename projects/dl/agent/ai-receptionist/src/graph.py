from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
import json

from state import ReceptionistState
from prompt import INTENT_PROMPT, EXTRACTION_PROMPT, FINAL_PROMPT
from tools import faq_lookup, book_appointment, send_message

llm = ChatOllama(model="llama3.2", temperature=0)


def classify_intent(state: ReceptionistState) -> ReceptionistState:
    prompt = PromptTemplate.from_template(INTENT_PROMPT)
    result = llm.invoke(prompt.format(input=state.get("user_input", "")))
    state["intent"] = (result.content or "").strip()
    print("Intent:", state["intent"])
    return state


def extract_fields(state: ReceptionistState) -> ReceptionistState:
    prompt = PromptTemplate.from_template(EXTRACTION_PROMPT)
    result = llm.invoke(prompt.format(input=state.get("user_input", "")))

    text = (result.content or "").strip()

    # Tolérant: si le modèle rajoute du texte
    try:
        start = text.find("{")
        end = text.rfind("}")
        data = json.loads(text[start:end + 1]) if start != -1 and end != -1 else {}
    except Exception:
        data = {}

    if data.get("name"):
        state["name"] = data["name"]
    if data.get("date_preference"):
        state["date_preference"] = data["date_preference"]
    if data.get("phone"):
        state["phone"] = data["phone"]
    if data.get("message"):
        state["message"] = data["message"]

    print("Extracted:", {k: state.get(k) for k in ["name", "date_preference", "phone", "message"]})
    return state


def ask_missing(state: ReceptionistState) -> ReceptionistState:
    intent = (state.get("intent") or "").strip()

    if intent == "BOOK_APPOINTMENT":
        if not state.get("name") and not state.get("date_preference"):
            state["response"] = "D’accord. Pour prendre le rendez-vous, quel est votre nom et quel créneau souhaitez-vous ? (ex: 01/04 matin)"
            return state
        if not state.get("name"):
            state["response"] = "Quel est votre nom, s’il vous plaît ?"
            return state
        if not state.get("date_preference"):
            state["response"] = "Pour quel jour/créneau souhaitez-vous le rendez-vous ? (ex: 01/04 matin)"
            return state

    if intent == "LEAVE_MESSAGE":
        if not state.get("name"):
            state["response"] = "Bien sûr. Quel est votre nom ?"
            return state
        if not state.get("phone"):
            state["response"] = "Quel est votre numéro de téléphone ?"
            return state
        if not state.get("message"):
            state["response"] = "Quel message voulez-vous laisser au praticien ?"
            return state

    state["response"] = None
    return state


def next_step_after_extract(state: ReceptionistState) -> str:
    intent = (state.get("intent") or "").strip()

    if intent == "BOOK_APPOINTMENT":
        return "ask_missing" if (not state.get("name") or not state.get("date_preference")) else "tool"

    if intent == "LEAVE_MESSAGE":
        missing = (not state.get("name")) or (not state.get("phone")) or (not state.get("message"))
        return "ask_missing" if missing else "tool"

    return "tool"  # ASK_INFO


def call_tool(state: ReceptionistState) -> ReceptionistState:
    intent = (state.get("intent") or "").strip()

    if intent == "ASK_INFO":
        result = faq_lookup(state.get("user_input", ""))

    elif intent == "BOOK_APPOINTMENT":
        name = state.get("name")
        date_pref = state.get("date_preference")
        if not name or not date_pref:
            state["tool_result"] = "MISSING_FIELDS"
            return state
        result = book_appointment(name, date_pref)

    elif intent == "LEAVE_MESSAGE":
        name = state.get("name")
        phone = state.get("phone")
        msg = state.get("message")
        if not name or not phone or not msg:
            state["tool_result"] = "MISSING_FIELDS"
            return state
        result = send_message(name, phone, msg)

    else:
        result = "Je peux aider pour un rendez-vous, une information (horaires/adresse/tarifs) ou laisser un message."

    state["tool_result"] = str(result)
    return state


def final_answer(state: ReceptionistState) -> ReceptionistState:
    if state.get("response"):
        return state

    if state.get("tool_result") == "MISSING_FIELDS":
        state["response"] = "Il me manque une information pour finaliser. Pouvez-vous préciser ?"
        return state

    prompt = PromptTemplate.from_template(FINAL_PROMPT)
    result = llm.invoke(prompt.format(
        intent=state.get("intent"),
        name=state.get("name"),
        date_preference=state.get("date_preference"),
        phone=state.get("phone"),
        message=state.get("message"),
        tool_result=state.get("tool_result"),
    ))

    state["response"] = (result.content or "").strip()
    return state


def build_graph():
    builder = StateGraph(ReceptionistState)

    builder.add_node("intent", classify_intent)
    builder.add_node("extract", extract_fields)
    builder.add_node("ask_missing", ask_missing)
    builder.add_node("tool", call_tool)
    builder.add_node("final", final_answer)

    builder.set_entry_point("intent")

    builder.add_edge("intent", "extract")

    builder.add_conditional_edges("extract", next_step_after_extract)

    builder.add_edge("ask_missing", END)

    builder.add_edge("tool", "final")
    builder.add_edge("final", END)

    return builder.compile()