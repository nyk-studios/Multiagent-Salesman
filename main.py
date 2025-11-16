# app_min3.py – Sparky Stage-1 with history + STATE handoff + clean exit
from __future__ import annotations

import os, json, re, yaml
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Any, Tuple, Sequence, Annotated

import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

DEFAULT_CONFIG_PATH = os.getenv("AGENT_CONFIG_PATH", "new_config.yaml")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY and ChatOpenAI)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))


class _SafeDict(dict):
    def __missing__(self, key):  # leave {unknown} as-is
        return "{" + key + "}"


def render_tmpl(tmpl: str, **ctx) -> str:
    return (tmpl or "").format_map(_SafeDict(**ctx))


def load_yaml_config(path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    sup = data.get("supervisor", {}) or {}
    for k in ("system", "human_template"):
        if not sup.get(k):
            raise ValueError(f"YAML supervisor must include '{k}'")
    supervisor_prompts = {"system": sup["system"], "human_template": sup["human_template"]}

    agents = data.get("agents", {})
    if not isinstance(agents, dict) or not agents:
        raise ValueError("YAML must include a top-level 'agents' mapping.")
    for key, cfg in agents.items():
        for req in ("name", "info_type", "system", "human_template"):
            if req not in cfg or not cfg[req]:
                raise ValueError(f"Agent '{key}' missing {req}")
    return agents, supervisor_prompts


@st.cache_resource(show_spinner=False)
def load_configs_cached():
    return load_yaml_config()


try:
    AGENT_CONFIGS, SUPERVISOR_PROMPTS = load_configs_cached()
except Exception as e:
    st.error(f"Failed to load YAML config: {e}")
    st.stop()


# ---------------- Reducer for stage_meta ----------------
def merge_stage_meta(left: Optional[Dict], right: Optional[Dict]) -> Dict:
    """Deep merge stage_meta dictionaries"""
    import copy
    result = copy.deepcopy(left) if left else {}
    if right:
        for key, value in right.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Deep merge nested dicts
                result[key] = {**result[key], **value}
            else:
                result[key] = value
    return result


# ---------------- Shared State ----------------
class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str
    collected_info: Dict[str, List[Any]]
    next_agent: str
    agent_index: int
    exchanges_with_current: int
    last_agent: str
    stage_meta: Annotated[Dict[str, Dict[str, Any]], merge_stage_meta]  #
    last_options: List[str]


# ---------------- Agent ----------------
class Agent:
    def __init__(self, name: str, system: str, human_template: str, info_type: str):
        self.name = name
        self.system = system
        self.human_template = human_template
        self.info_type = info_type

    def _mock_response(self, user_text: str, last_agent: str = "") -> str:
        return (
            "Hey there! I'm Sparky – quick check-in before we explore together.\n"
            "What feels most true right now?\n"
            "A) I want to restart my creative spark\n"
            "B) I'm curious but hesitant\n"
            "C) I'm overwhelmed and need calm\n"
            "D) Not sure yet\n\n"
            "[INTENT_TYPE: Exploratory]\n"
            "[CONFIRM_INTENT: partial]\n"
            "[EMO_TONE: neutral]\n"
            "[BEHAVIORAL_SIGNAL: explorer]\n"
            "STATE={\"last_theme\":\"exploration\",\"last_level\":\"L1\",\"turn\":1}"
        )

    def _generate_response(self, history: str, user_text: str, state: AgentState) -> str:
        if not USE_OPENAI:
            return self._mock_response(user_text, state.get("last_agent", ""))

        # Pull last STATE for this agent (theme/level/turn) to feed back
        last_state_json = state.get("stage_meta", {}).get(self.info_type, {}).get("last_state_json", {})
        turns_for_info = len(state.get("collected_info", {}).get(self.info_type, []))
        last_intent_status = state.get("stage_meta", {}).get(self.info_type, {}).get("intent_status", "")

        ctx = {
            "history": history,  # human-readable transcript (last ~8 msgs below)
            "user_text": user_text,
            "last_agent": state.get("last_agent", "conversation"),
            "turns_for_info": str(turns_for_info),
            "last_intent": last_intent_status,
            "collected_info": json.dumps(state.get("collected_info", {}), ensure_ascii=False),
            "state_json": json.dumps(last_state_json, ensure_ascii=False),
        }

        sys_content = render_tmpl(self.system, **ctx)
        human_content = render_tmpl(self.human_template, **ctx)

        #  Include prior conversation so the LLM sees last Q&A
        recent_msgs = list(state.get("messages", []))[-8:]
        msgs: List[BaseMessage] = [SystemMessage(content=sys_content)] + recent_msgs + [
            HumanMessage(content=human_content)]

        try:
            llm = ChatOpenAI(model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE)
            out = llm.invoke(msgs)
            return (out.content or "").strip() or self._mock_response(user_text)
        except Exception:
            return self._mock_response(user_text)

    def process(self, state: AgentState) -> AgentState:
        # Build readable history for template (still send true messages to LLM above)
        last_msgs = list(state.get("messages", []))[-8:]
        history = "\n".join([getattr(m, "content", "") for m in last_msgs])
        user_text = state.get("user_input", "")

        reply = self._generate_response(history, user_text, state)
        print("\n\n====== RAW LLM REPLY ======\n", reply, "\n===========================\n\n")

        # -------- Parse Tags --------
        def _tag(pattern, default=None, cast=lambda x: x):
            m = re.search(pattern, reply, re.I)
            return cast(m.group(1)) if m else default

        intent_status = _tag(r"\[CONFIRM_INTENT:\s*(clear|partial|unclear)\s*\]", "partial")
        intent_type = _tag(
            r"\[INTENT_TYPE:\s*(Exploratory|Aspirational|Therapeutic|Achievement|Connection|unknown)\s*\]", "unknown")
        emotional_tone = _tag(r"\[EMO_TONE:\s*(positive|neutral|tense|resistant)\s*\]", "neutral")
        behavioral_signal = _tag(r"\[BEHAVIORAL_SIGNAL:\s*(explorer|planner|reflector|hands_on)\s*\]", "explorer")

        # Parse STATE JSON block
        state_match = re.search(r"STATE\s*=\s*(\{.*\})", reply, re.S)
        state_payload: Dict[str, Any] = {}
        if state_match:
            try:
                state_payload = json.loads(state_match.group(1))
            except Exception:
                state_payload = {}

        # Clean tags/STATE from user-facing text
        clean_for_user = re.sub(r"\n?STATE\s*=\s*\{.*\}\s*$", "", reply, flags=re.S).strip()
        clean_for_user = re.sub(r"\n?\[[A-Za-z_]+:\s*[^\]]+\]\s*", "", clean_for_user).strip()

        #  Create a NEW stage_meta dictionary with deep copy
        import copy
        new_stage_meta = copy.deepcopy(state.get("stage_meta", {}))

        #  Replace entire sub-dict (not update)
        new_stage_meta[self.info_type] = {
            "intent_status": intent_status,  # clear | partial | unclear
            "intent_type": intent_type,  # Exploratory | Aspirational | ...
            "emotional_tone": emotional_tone,  # positive | neutral | tense | resistant
            "behavioral_signal": behavioral_signal,  # explorer | planner | reflector | hands_on
            "last_state_json": state_payload,  # tiny theme/level/turn memory
        }

        #  DEBUG: Print parsed values
        print(f"DEBUG - Parsed intent_status: {intent_status}")
        print(f"DEBUG - Parsed intent_type: {intent_type}")
        print(f"DEBUG - Parsed emotional_tone: {emotional_tone}")
        print(f"DEBUG - Parsed behavioral_signal: {behavioral_signal}")
        print(f"DEBUG - New stage_meta: {new_stage_meta}")

        # collected info for sidebar
        merged_collected = dict(state.get("collected_info", {}))
        if user_text:
            merged_collected.setdefault(self.info_type, []).append(user_text)

        #  Return the NEW dictionary
        updates: AgentState = {
            "messages": [AIMessage(content=clean_for_user)],
            "collected_info": merged_collected,
            "stage_meta": new_stage_meta,
            "exchanges_with_current": state.get("exchanges_with_current", 0) + 1,
            "last_agent": self.info_type,
        }

        print(f"DEBUG - Returning stage_meta: {updates['stage_meta']}")
        return updates


# ---------------- Supervisor ----------------
class Supervisor:
    def __init__(self, agents: Dict[str, Agent], prompts: Dict[str, str]):
        self.agents = agents
        self.agent_keys = list(agents.keys())
        self.system = prompts["system"]
        self.human_template = prompts["human_template"]
        self.use_openai = USE_OPENAI and bool(self.system and self.human_template)

    def route(self, state: AgentState) -> AgentState:
        collected = state.get("collected_info", {})
        meta_conn = state.get("stage_meta", {}).get("connection", {})
        conn_turns = len(collected.get("connection", []))

        intent_status = (meta_conn.get("intent_status") or "").lower()
        emotional_tone = (meta_conn.get("emotional_tone") or "").lower()

        print(
            f"DEBUG SUPERVISOR - conn_turns: {conn_turns}, intent_status: {intent_status}, emotional_tone: {emotional_tone}")

        #  Exit when confirmed OR after 3 questions as a safety stop
        ready = (intent_status == "clear") and (emotional_tone != "resistant")
        if ready or conn_turns >= 3:
            print(f"DEBUG SUPERVISOR - FINISHING (ready={ready}, conn_turns={conn_turns})")
            return {"next_agent": "FINISH", "exchanges_with_current": 0}

        # default: keep in connection stage
        if "connection" in self.agent_keys:
            print(f"DEBUG SUPERVISOR - Continuing with connection agent")
            return {"agent_index": self.agent_keys.index("connection"),
                    "next_agent": "connection", "exchanges_with_current": 0}
        return {"next_agent": "FINISH", "exchanges_with_current": 0}


# ---------------- Graph ----------------
def create_graph(agent_configs: Dict[str, Dict[str, str]]):
    agents = {k: Agent(cfg["name"], cfg["system"], cfg["human_template"], cfg["info_type"])
              for k, cfg in agent_configs.items()}
    supervisor = Supervisor(agents, prompts=SUPERVISOR_PROMPTS)

    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor.route)

    for k, a in agents.items():
        workflow.add_node(k, a.process)

    main_keys = list(agents.keys())

    def route_from_supervisor(state: AgentState) -> str:
        nxt = (state.get("next_agent") or "").lower()
        if nxt in ("finish", "end"):
            return "END"
        return nxt if nxt in main_keys else (main_keys[0] if main_keys else "END")

    edges_map = {k: k for k in main_keys}
    edges_map.update({"END": END})
    workflow.add_conditional_edges("supervisor", route_from_supervisor, edges_map)

    for k in main_keys:
        workflow.add_edge(k, END)

    workflow.set_entry_point("supervisor")
    return workflow.compile()


# ---------------- Helper: extract options from assistant text ----------------
def extract_options(text: str) -> Tuple[str, List[str]]:
    """
    Find lines like 'A) something', 'B) something else' and turn them into options.
    Returns (clean_text_without_option_lines, options_list).
    """
    lines = text.splitlines()
    opt_pattern = re.compile(r"^\s*([A-Z])\)\s*(.+)")  # e.g. 'A) I want to restart...'
    options: List[str] = []
    kept_lines: List[str] = []

    for line in lines:
        m = opt_pattern.match(line)
        if m:
            options.append(m.group(2).strip())
        else:
            kept_lines.append(line)

    clean_text = "\n".join(kept_lines).strip()
    return clean_text, options


# ---------------- Turn execution ----------------
def process_user_message(graph, state: AgentState, msg: str) -> Tuple[str, AgentState]:
    if (state.get("next_agent") or "").upper() == "FINISH":
        # No more questions: clear options
        state["last_options"] = []
        return "We've collected everything we need. Thanks!", state

    print(f"DEBUG - Input state stage_meta: {state.get('stage_meta', {})}")

    before = len(state.get("messages", []))
    state = {
        **state,
        "messages": list(state.get("messages", [])) + [HumanMessage(content=msg)],
        "user_input": msg
    }
    new_state = graph.invoke(state)

    print(f"DEBUG - Output state stage_meta: {new_state.get('stage_meta', {})}")

    after = new_state.get("messages", [])
    ai_msgs = [m for m in after[before:] if isinstance(m, AIMessage)]
    raw_ai_text = "\n\n".join([m.content for m in ai_msgs if m.content])

    # NEW: split assistant reply into visible text + button options
    clean_text, options = extract_options(raw_ai_text)
    new_state["last_options"] = options

    return clean_text, new_state


# ---------------- Streamlit UI ----------------
def _init_session():
    if "graph" not in st.session_state:
        st.session_state.graph = create_graph(AGENT_CONFIGS)
    if "state" not in st.session_state:
        st.session_state.state = AgentState(
            messages=[],
            collected_info={k: [] for k in {cfg["info_type"] for cfg in AGENT_CONFIGS.values()}},
            next_agent="", agent_index=0, exchanges_with_current=0, last_agent="",
            stage_meta={},
            last_options=[]
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def main():
    st.set_page_config(page_title="Sparky – Stage 1 (Connection & Curiosity)", layout="wide")
    st.title("🤖 Sparky – Stage 1 (Connection & Curiosity)")
    _init_session()
    st.info("OpenAI active" if USE_OPENAI else "Mock mode", icon="✅" if USE_OPENAI else "⚠️")

    # Auto-start: produce first assistant message on load
    if not st.session_state.chat_history:
        first_state = st.session_state.state
        first_state["user_input"] = ""  # empty first turn
        ai_text, new_state = process_user_message(st.session_state.graph, first_state, "")
        st.session_state.state = new_state
        st.session_state.chat_history.append({"role": "assistant", "content": ai_text})

    with st.sidebar:
        st.header("Collected Info")
        for k, v in st.session_state.state.get("collected_info", {}).items():
            st.markdown(f"**{k}** ({len(v)})")
            for i, it in enumerate(v[-5:]):
                st.write(f"- {i + 1}. {it}")

        st.markdown("---")
        st.header("Conversation Insights")
        meta = st.session_state.state.get("stage_meta", {})
        conn = (meta or {}).get("connection", {})
        st.markdown(f"- **Confirm Intent:** {conn.get('intent_status', '—')}")
        st.markdown(f"- **Intent Type:** {conn.get('intent_type', '—')}")
        st.markdown(f"- **Emotional Tone:** {conn.get('emotional_tone', '—')}")
        st.markdown(f"- **Behavioral Signal:** {conn.get('behavioral_signal', '—')}")
        st.markdown(f"- **STATE:** {json.dumps(conn.get('last_state_json', {}), ensure_ascii=False)}")

        if st.button("Reset Conversation", type="secondary"):
            keep = {"graph"}
            for k in list(st.session_state.keys()):
                if k not in keep:
                    del st.session_state[k]
            _init_session();
            st.rerun()

    # Show chat history
    for t in st.session_state.chat_history:
        with st.chat_message(t["role"]):
            st.markdown(t["content"])

    # ---- Buttons for current options (if any) ----
    options = st.session_state.state.get("last_options", []) or []
    if options and st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write("Choose an option:")
            cols = st.columns(len(options))
            for i, opt in enumerate(options):
                if cols[i].button(opt, key=f"opt_{i}"):
                    # Treat button click as if the user typed that option
                    st.session_state.chat_history.append({"role": "user", "content": opt})
                    ai_text, new_state = process_user_message(
                        st.session_state.graph, st.session_state.state, opt
                    )
                    st.session_state.state = new_state
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_text})
                    st.rerun()

    # Free-text input is still available (optional)
    prompt = st.chat_input("Type your message...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        ai_text, new_state = process_user_message(st.session_state.graph, st.session_state.state, prompt)
        st.session_state.state = new_state
        st.session_state.chat_history.append({"role": "assistant", "content": ai_text})
        with st.chat_message("assistant"):
            st.markdown(ai_text)
        st.rerun()


if __name__ == "__main__":
    main()