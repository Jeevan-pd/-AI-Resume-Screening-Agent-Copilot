"""
AI Agent Copilot Module for Resume Screening Agent.
Implements an interactive agentic reasoning loop with tools for document reading,
candidate lookup, database persistence (SQLite), and interview scheduling.
"""

import os
import json
import sqlite3
import logging
from typing import Dict, Any, List, Tuple
from config import GROQ_API_KEY, OPENAI_API_KEY, MODEL_PROVIDER, GROQ_MODEL, OPENAI_MODEL, BASE_DIR

logger = logging.getLogger(__name__)

# Try importing LLM clients
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# SQLite Database Initialization for Candidate Tracking & Notes
DB_PATH = BASE_DIR / "output" / "candidate_tracking.db"

def init_db():
    """Initialize local SQLite database for candidate tracking."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidate_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT UNIQUE,
            status TEXT,
            notes TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interview_schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT,
            interview_date TEXT,
            interviewer TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()


class RecruiterAIAgent:
    """Conversational AI Recruiter Agent with Reasoning, Tool Use, and Database Actions."""

    def __init__(self, api_key: str = None, provider: str = None):
        self.api_key = api_key or GROQ_API_KEY or OPENAI_API_KEY
        self.provider = provider or MODEL_PROVIDER
        self.groq_client = None
        self.openai_client = None

        if self.provider == "groq" and GROQ_AVAILABLE and self.api_key:
            try:
                self.groq_client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Groq init error: {e}")

        if self.provider == "openai" and OPENAI_AVAILABLE and self.api_key:
            try:
                self.openai_client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"OpenAI init error: {e}")

    # ==========================================
    # AGENT TOOL EXECUTIONS
    # ==========================================

    @staticmethod
    def tool_search_candidates(candidates_data: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Tool: Search candidate list by name, skill, score, or recommendation tier."""
        q_lower = query.lower()
        matched = []
        for c in candidates_data:
            name = c.get("candidate_name", "").lower()
            rec = c.get("recommendation", "").lower()
            skills = [s.lower() for s in c.get("matching_skills", [])]
            
            if q_lower in name or q_lower in rec or any(q_lower in s for s in skills):
                matched.append({
                    "name": c.get("candidate_name"),
                    "score": c.get("overall_score"),
                    "recommendation": c.get("recommendation"),
                    "matching_skills": c.get("matching_skills"),
                    "email": c.get("email")
                })
        return matched if matched else candidates_data[:3]

    @staticmethod
    def tool_read_resume(candidates_data: List[Dict[str, Any]], candidate_name: str) -> str:
        """Tool: Retrieve full details and summaries of a specific candidate."""
        for c in candidates_data:
            if candidate_name.lower() in c.get("candidate_name", "").lower():
                return f"""
Candidate Details for {c.get('candidate_name')}:
- Overall Score: {c.get('overall_score')}/100 ({c.get('recommendation')})
- Contact: {c.get('email')} | {c.get('phone')}
- Matched Skills: {', '.join(c.get('matching_skills', []))}
- Missing Skills: {', '.join(c.get('missing_skills', []))}
- Experience Summary: {c.get('experience_summary')}
- Education Summary: {c.get('education_summary')}
- Strengths: {'; '.join(c.get('strengths', []))}
                """.strip()
        return f"Candidate '{candidate_name}' not found in screened candidate pool."

    @staticmethod
    def tool_save_candidate_status(candidate_name: str, status: str, notes: str = "") -> str:
        """Tool: Save or update candidate status in local SQLite database (e.g. Shortlisted, Rejected, Interview Scheduled)."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO candidate_status (candidate_name, status, notes)
                VALUES (?, ?, ?)
                ON CONFLICT(candidate_name) DO UPDATE SET
                    status=excluded.status,
                    notes=excluded.notes,
                    updated_at=CURRENT_TIMESTAMP
            """, (candidate_name, status, notes))
            conn.commit()
            conn.close()
            return f"✅ Candidate '{candidate_name}' status successfully updated to '{status}' in database."
        except Exception as e:
            return f"Failed to update database status: {e}"

    @staticmethod
    def tool_schedule_interview(candidate_name: str, date_time: str, interviewer: str = "Hiring Manager", notes: str = "") -> str:
        """Tool: Schedule an interview for a candidate and save to database."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO interview_schedules (candidate_name, interview_date, interviewer, notes)
                VALUES (?, ?, ?, ?)
            """, (candidate_name, date_time, interviewer, notes))
            conn.commit()
            conn.close()
            return f"📅 Interview scheduled for '{candidate_name}' on {date_time} with {interviewer}."
        except Exception as e:
            return f"Failed to schedule interview: {e}"

    @staticmethod
    def tool_get_database_status() -> List[Dict[str, Any]]:
        """Tool: Fetch all candidate status records from SQLite database."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT candidate_name, status, notes, updated_at FROM candidate_status ORDER BY updated_at DESC")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    # ==========================================
    # REASONING & RESPONSE GENERATION LOOP
    # ==========================================

    def process_query(
        self,
        user_query: str,
        candidates_data: List[Dict[str, Any]],
        jd_text: str = "",
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Processes user query through reasoning loop:
        1. Understand User Intent
        2. Execute Tools (Lookup candidates, read resumes, query/update database)
        3. Formulate useful response
        """
        actions_taken = []
        q_lower = user_query.lower()

        # Action Execution Detection
        # 1. Action: Save / Update Candidate Status
        if any(w in q_lower for w in ["shortlist", "reject", "save status", "mark as", "hire"]):
            for c in candidates_data:
                c_name = c.get("candidate_name", "")
                if c_name.lower() in q_lower:
                    if "shortlist" in q_lower or "hire" in q_lower or "recommend" in q_lower:
                        status = "Shortlisted"
                    elif "reject" in q_lower:
                        status = "Rejected"
                    else:
                        status = "Under Review"
                    
                    res_msg = self.tool_save_candidate_status(c_name, status, f"Updated via AI Agent chat query: '{user_query}'")
                    actions_taken.append(f"🛠️ Executed Database Action: {res_msg}")

        # 2. Action: Schedule Interview
        if "schedule" in q_lower or "interview date" in q_lower:
            for c in candidates_data:
                c_name = c.get("candidate_name", "")
                if c_name.lower() in q_lower:
                    res_msg = self.tool_schedule_interview(c_name, "Tomorrow at 10:00 AM", "Lead AI Engineer", "Initial technical round")
                    actions_taken.append(f"🛠️ Executed Scheduling Action: {res_msg}")

        # Context Prep
        cand_summary_list = []
        for c in candidates_data[:10]:
            cand_summary_list.append(f"- Name: {c.get('candidate_name')}, Score: {c.get('overall_score')}/100, Tier: {c.get('recommendation')}, Matched Skills: {', '.join(c.get('matching_skills', []))}, Missing Skills: {', '.join(c.get('missing_skills', []))}")

        cand_context = "\n".join(cand_summary_list)
        db_history = self.tool_get_database_status()
        db_context = json.dumps(db_history) if db_history else "No database records yet."

        prompt = f"""
        You are an intelligent AI Recruitment Assistant & Copilot.
        You take natural language commands from the recruiter, analyze candidate data, perform lookups/actions, and return a useful, professional answer.

        Current Candidate Pool Context:
        {cand_context}

        Database Status History:
        {db_context}

        User Request: "{user_query}"

        Instructions:
        - Analyze the user request.
        - Answer directly, clearly, and insightfully.
        - If the user requested an action (like shortlisting, reading details, comparing candidates, scheduling, or skill gap analysis), confirm the action and provide clear conclusions.
        - Format output with clean Markdown formatting, bullet points, and key takeaways.
        """

        # LLM Call
        response_text = ""
        if self.groq_client:
            try:
                res = self.groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1000
                )
                response_text = res.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq API error in agent: {e}")

        if not response_text and self.openai_client:
            try:
                res = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1000
                )
                response_text = res.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API error in agent: {e}")

        # Rule-based fallback response
        if not response_text:
            response_text = f"Here is the evaluation for your query on **{candidates_data[0].get('candidate_name', 'Candidates')}**:\n\n"
            for c in candidates_data[:3]:
                response_text += f"- **{c.get('candidate_name')}** ({c.get('overall_score')}/100 - {c.get('recommendation')}): Matched {len(c.get('matching_skills', []))} skills ({', '.join(c.get('matching_skills', [])[:4])}).\n"

        return {
            "response": response_text,
            "actions_taken": actions_taken
        }
