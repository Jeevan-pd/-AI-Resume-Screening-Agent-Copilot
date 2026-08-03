"""
AI Recruiter Assistant Module for Resume Screening Agent.
Implements an interactive assistant with candidate lookups, SQLite database tracking, and interview scheduling.
100% self-contained and offline-capable without external API keys.
"""

import sqlite3
import logging
from typing import Dict, Any, List
from config import BASE_DIR

logger = logging.getLogger(__name__)

# SQLite Database Path
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
    """Conversational AI Recruiter Assistant with Tool Execution and Database Actions."""

    @staticmethod
    def tool_save_candidate_status(candidate_name: str, status: str, notes: str = "") -> str:
        """Save or update candidate status in local SQLite database."""
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
        """Schedule an interview for a candidate and save to database."""
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
        """Fetch all candidate status records from SQLite database."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT candidate_name, status, notes, updated_at FROM candidate_status ORDER BY updated_at DESC")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def process_query(
        self,
        user_query: str,
        candidates_data: List[Dict[str, Any]],
        jd_text: str = "",
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Processes user query and performs lookups/actions."""
        actions_taken = []
        q_lower = user_query.lower()

        # Action: Shortlist / Update Status
        if any(w in q_lower for w in ["shortlist", "reject", "save status", "mark as", "hire"]):
            for c in candidates_data:
                c_name = c.get("candidate_name", "")
                if c_name.lower() in q_lower:
                    if "shortlist" in q_lower or "hire" in q_lower:
                        status = "Shortlisted"
                    elif "reject" in q_lower:
                        status = "Rejected"
                    else:
                        status = "Under Review"
                    
                    res_msg = self.tool_save_candidate_status(c_name, status, f"Updated via Recruiter Assistant query: '{user_query}'")
                    actions_taken.append(f"🛠️ Executed Database Action: {res_msg}")

        # Action: Schedule Interview
        if "schedule" in q_lower or "interview" in q_lower:
            for c in candidates_data:
                c_name = c.get("candidate_name", "")
                if c_name.lower() in q_lower:
                    res_msg = self.tool_schedule_interview(c_name, "Tomorrow at 10:00 AM", "Hiring Lead", "Technical Screening")
                    actions_taken.append(f"🛠️ Executed Scheduling Action: {res_msg}")

        # Formulate Response
        if not candidates_data:
            return {
                "response": "No candidates screened yet. Please upload candidate resumes or load the demo dataset.",
                "actions_taken": actions_taken
            }

        top_cand = candidates_data[0]
        
        if "top candidate" in q_lower or "best" in q_lower or "who" in q_lower:
            reply = (
                f"### 🏆 Top Candidate Analysis\n\n"
                f"The top-ranked candidate is **{top_cand['candidate_name']}** with an overall score of **{top_cand['overall_score']}/100** ({top_cand['recommendation']}).\n\n"
                f"- **Matched Skills**: {', '.join(top_cand.get('matching_skills', []))}\n"
                f"- **Strengths**: {'; '.join(top_cand.get('strengths', []))}\n"
                f"- **Experience**: {top_cand.get('experience_summary')}"
            )
        elif "database" in q_lower or "records" in q_lower or "status" in q_lower:
            records = self.tool_get_database_status()
            if records:
                rec_str = "\n".join([f"- **{r['candidate_name']}**: Status `{r['status']}` ({r['notes']})" for r in records])
                reply = f"### 💾 Candidate Database Records\n\n{rec_str}"
            else:
                reply = "No saved database records found yet. Instruct me to e.g. *'Shortlist Alex Rivera'* to save candidate status!"
        else:
            reply = f"Here is the summary of candidate evaluation results:\n\n"
            for c in candidates_data[:4]:
                reply += f"- **{c.get('candidate_name')}**: Score **{c.get('overall_score')}/100** ({c.get('recommendation')}) | Matched: {', '.join(c.get('matching_skills', [])[:4])}\n"

        return {
            "response": reply,
            "actions_taken": actions_taken
        }
