"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid
import os

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_path, db_conn, replay_events
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("🚀 HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
async def get_recent_threads():
    async with db_conn() as db:
        async with db.execute("""
            SELECT thread_id, pr_url, 
                   MIN(timestamp) as started, 
                   MAX(timestamp) as last_event,
                   COUNT(*) as events
            FROM audit_events
            GROUP BY thread_id
            ORDER BY last_event DESC
            LIMIT 10
        """) as cursor:
            return await cursor.fetchall()

async def get_stats():
    async with db_conn() as db:
        async with db.execute("""
            SELECT AVG(confidence) as avg_conf,
                   COUNT(DISTINCT thread_id) as total_sessions,
                   SUM(CASE WHEN decision = 'approve' OR decision = 'auto' THEN 1 ELSE 0 END) as approved
            FROM audit_events
            WHERE action IN ('route', 'human_approval', 'auto_approve')
        """) as cursor:
            return await cursor.fetchone()

with st.sidebar:
    st.header("📊 Confidence Calibration")
    stats = asyncio.run(get_stats())
    if stats and stats['total_sessions'] > 0:
        col1, col2 = st.columns(2)
        col1.metric("Avg Confidence", f"{stats['avg_conf']:.0%}")
        col2.metric("Approval Rate", f"{(stats['approved'] or 0)/stats['total_sessions']:.0%}")
        st.progress(stats['avg_conf'] or 0.0, text="AI Certainty Level")
    else:
        st.caption("No stats yet. Run some reviews!")

    st.divider()
    st.header("🕒 Recent sessions")
    threads = asyncio.run(get_recent_threads())
    for t in threads:
        with st.container(border=True):
            st.write(f"**PR:** {t['pr_url'].split('/')[-1]}")
            st.caption(f"ID: `{t['thread_id'][:8]}...`")
            if st.button("Resume / View", key=f"btn_{t['thread_id']}"):
                st.session_state.thread_id = t['thread_id']
                st.session_state.pr_url = t['pr_url']
                st.session_state.interrupt_payload = None
                st.session_state.final = None
                st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url_input = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.success(f"### ✅ Approval requested — confidence {conf:.0%}")
    st.info(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    st.write("#### 📝 Proposed Comments")
    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("🔍 View Full Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Add feedback or instructions (optional)", key="approval_feedback")
    
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Request Changes (Edit)", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.warning(f"### ⚠️ Strong escalation — confidence {conf:.0%}")
    st.info(payload["confidence_reasoning"])
    
    if payload.get("risk_factors"):
        st.error("**Risks identified:**\n" + "\n".join([f"- {r}" for r in payload["risk_factors"]]))
    
    st.markdown(payload["summary"])

    with st.form("escalation_form"):
        st.write("#### ❓ Clarifying Questions")
        st.write("The agent needs more context to complete the review:")
        answers = {}
        for i, q in enumerate(payload["questions"]):
            answers[q] = st.text_area(f"Q{i+1}: {q}", key=f"q_{i}")
        
        if st.form_submit_button("Submit Answers", type="primary"):
            return answers
    return None


def render_timeline(thread_id: str):
    """Render the audit trail as a visual timeline."""
    st.divider()
    st.write("### 📜 Session Timeline")
    events = asyncio.run(replay_events(thread_id))
    
    if not events:
        st.caption("No events logged yet.")
        return

    for e in events:
        with st.container(border=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.caption(e["timestamp"].split("T")[1][:8])
                st.write(f"**{e['action'].replace('_', ' ').upper()}**")
            with col2:
                risk_color = {"low": "green", "med": "orange", "high": "red"}.get(e["risk_level"], "grey")
                st.markdown(f"Risk: :{risk_color}[{e['risk_level']}] | Confidence: **{e['confidence']:.0%}**")
                st.write(e["reason"])
                if e["reviewer_id"]:
                    st.caption(f"👤 Reviewer: {e['reviewer_id']}")


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            # Check if there is an existing state for this thread
            state = await app.aget_state(cfg)
            if state.values:
                # If it exists and has an interrupt, just return that
                if state.next:
                    result = state.values
                    # Add back the interrupt marker for the main loop
                    result["__interrupt__"] = state.tasks[0].interrupts if state.tasks else []
                    return result
                return state.values
            
            # Start fresh
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url_input:
    st.session_state.pr_url = pr_url_input
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.rerun()

if st.session_state.thread_id:
    # Always try to fetch current state
    with st.spinner("Processing..."):
        result = asyncio.run(run_graph(st.session_state.pr_url, st.session_state.thread_id))
    
    if "__interrupt__" in result and result["__interrupt__"]:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming graph..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result and result["__interrupt__"]:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.balloons()
        st.success(f"### ✅ {action.replace('_', ' ').title()}!")
        st.write(f"The review comment has been successfully posted to: {st.session_state.pr_url}")
    elif action == "rejected":
        st.warning("### 🚫 Review Rejected")
        st.write("No comment was posted to GitHub as per your request.")
    else:
        st.info(f"### Final state: {action}")
    
    st.divider()
    st.caption(f"**Thread ID:** `{st.session_state.thread_id}`")

# Always show timeline if we have a thread
if st.session_state.thread_id:
    render_timeline(st.session_state.thread_id)
