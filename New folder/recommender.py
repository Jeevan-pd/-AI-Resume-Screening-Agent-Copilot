"""
Recommender Module for AI Resume Screening Agent.
Generates candidate recommendations, qualitative evaluation reports, strengths/weaknesses,
and LLM-driven features (AI Interview Questions, Score Explanations, Skill Gap Analysis).
"""

import json
import logging
from typing import Dict, Any, List

from config import RECOMMENDATION_THRESHOLDS, GROQ_API_KEY, OPENAI_API_KEY, MODEL_PROVIDER, GROQ_MODEL, OPENAI_MODEL

logger = logging.getLogger(__name__)

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


class CandidateRecommender:
    """Generates candidate recommendations, insights, and LLM-powered interview kits."""

    def __init__(self, api_key: str = None, provider: str = None):
        self.api_key = api_key or GROQ_API_KEY or OPENAI_API_KEY
        self.provider = provider or MODEL_PROVIDER
        self.groq_client = None
        self.openai_client = None

        if self.provider == "groq" and GROQ_AVAILABLE and self.api_key:
            try:
                self.groq_client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Groq client init error in Recommender: {e}")

        if self.provider == "openai" and OPENAI_AVAILABLE and self.api_key:
            try:
                self.openai_client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"OpenAI client init error in Recommender: {e}")

    @staticmethod
    def get_recommendation_category(score: float) -> str:
        """Classifies total score into recommendation tiers."""
        if score >= RECOMMENDATION_THRESHOLDS["Highly Recommended"]:
            return "Highly Recommended"
        elif score >= RECOMMENDATION_THRESHOLDS["Recommended"]:
            return "Recommended"
        elif score >= RECOMMENDATION_THRESHOLDS["Consider"]:
            return "Consider"
        else:
            return "Not Recommended"

    @classmethod
    def generate_report(
        cls,
        candidate_info: Dict[str, Any],
        jd_info: Dict[str, Any],
        score_eval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generates structured evaluation report for a candidate."""
        overall_score = score_eval["overall_score"]
        sub_scores = score_eval["sub_scores"]
        matched_skills = score_eval["matched_skills"]
        missing_skills = score_eval["missing_skills"]

        recommendation = cls.get_recommendation_category(overall_score)

        # Build Strengths
        strengths = []
        if sub_scores["skill_score"] >= 80:
            strengths.append(f"Strong technical alignment with {len(matched_skills)} key skills matched.")
        if sub_scores["experience_score"] >= 85:
            strengths.append(f"Solid experience record ({candidate_info.get('experience_years', 0)} years vs {jd_info.get('experience_required_years', 0)} required).")
        if sub_scores["semantic_score"] >= 75:
            strengths.append("High context and semantic correlation between resume project descriptions and target role.")
        if sub_scores["education_score"] >= 90:
            strengths.append("Meets or exceeds target education credentials.")
        if not strengths:
            strengths.append("Possesses foundational qualifications relevant to the role.")

        # Build Weaknesses
        weaknesses = []
        if missing_skills:
            weaknesses.append(f"Lacks key required skills: {', '.join(missing_skills[:4])}.")
        if sub_scores["experience_score"] < 70:
            weaknesses.append(f"Experience gap ({candidate_info.get('experience_years', 0)} yrs vs {jd_info.get('experience_required_years', 0)} yrs requested).")
        if sub_scores["semantic_score"] < 60:
            weaknesses.append("Resume phrasing indicates potential gap in domain-specific terminology.")
        if not weaknesses:
            weaknesses.append("No critical deal-breaker gaps identified.")

        # Summaries
        exp_summary = f"{candidate_info.get('name', 'Candidate')} has approximately {candidate_info.get('experience_years', 0)} years of relevant industry experience."
        edu_list = candidate_info.get('education', ['Degree Not Stated'])
        edu_summary = f"Education Background: {', '.join(edu_list)}."

        return {
            "candidate_name": candidate_info.get("name", "Candidate"),
            "email": candidate_info.get("email", "N/A"),
            "phone": candidate_info.get("phone", "N/A"),
            "overall_score": overall_score,
            "recommendation": recommendation,
            "sub_scores": sub_scores,
            "matching_skills": matched_skills,
            "missing_skills": missing_skills,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "experience_summary": exp_summary,
            "education_summary": edu_summary,
        }

    # ==========================================
    # LLM-POWERED BONUS FEATURES
    # ==========================================

    def _call_llm(self, prompt: str) -> str:
        """Execute LLM call."""
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq API call error: {e}")

        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API call error: {e}")

        return ""

    def generate_interview_questions(self, candidate_name: str, jd_title: str, matched_skills: List[str], missing_skills: List[str]) -> List[str]:
        """AI-generated tailored interview questions targeting matched strengths and missing skill gaps."""
        prompt = f"""
        You are a Senior Technical Recruiter. Generate 5 targeted interview questions for candidate '{candidate_name}' applying for '{jd_title}'.
        
        Matched Skills: {', '.join(matched_skills[:5]) if matched_skills else 'General'}
        Missing/Gap Skills: {', '.join(missing_skills[:5]) if missing_skills else 'None'}
        
        Provide 5 clear, insightful technical and behavioral interview questions in a clean numbered list.
        """
        response = self._call_llm(prompt)
        if response:
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            questions = [l for l in lines if l[0].isdigit() or l.startswith("-") or l.startswith("*")]
            return questions if questions else lines[:5]

        # Fallback preset questions
        return [
            f"1. Can you walk us through a recent project where you utilized {matched_skills[0] if matched_skills else 'your core technical skills'}?",
            f"2. How would you handle a production challenge involving {missing_skills[0] if missing_skills else 'a technology you are less familiar with'}?",
            "3. Describe a time you had to balance tight deadlines with code quality and maintainability.",
            "4. How do you approach learning and integrating new tools or frameworks into an existing stack?",
            "5. What architecture patterns do you follow when designing scalable, reliable software systems?"
        ]

    def explain_score(self, candidate_name: str, score_eval: Dict[str, Any]) -> str:
        """Generates an executive natural-language explanation of how the candidate's score was derived."""
        sub = score_eval["sub_scores"]
        overall = score_eval["overall_score"]
        
        prompt = f"""
        Write a concise, professional executive explanation (2 paragraphs) detailing how {candidate_name}'s screening score of {overall}/100 was calculated.

        Sub-scores:
        - Skill Match (40% weight): {sub['skill_score']}%
        - Experience Match (30% weight): {sub['experience_score']}%
        - Education Match (15% weight): {sub['education_score']}%
        - Semantic Similarity (15% weight): {sub['semantic_score']}%
        """
        response = self._call_llm(prompt)
        if response:
            return response.strip()

        # Fallback template
        return (
            f"Candidate **{candidate_name}** achieved an overall score of **{overall}/100** based on our weighted evaluation model. "
            f"The score reflects strong skill alignment ({sub['skill_score']}% match) and an experience match score of {sub['experience_score']}%. "
            f"Education match contributed {sub['education_score']}%, while contextual semantic vector analysis measured a similarity of {sub['semantic_score']}%."
        )

    def analyze_skill_gap(self, candidate_name: str, missing_skills: List[str], required_skills: List[str]) -> str:
        """Generates actionable skill gap analysis and recommended onboarding/upskilling plan."""
        if not missing_skills:
            return f"**{candidate_name}** possesses all required skills! No critical skill gaps identified."

        prompt = f"""
        Provide a concise skill gap analysis for candidate {candidate_name}.
        Missing Required Skills: {', '.join(missing_skills)}
        
        Provide 3 strategic recommendations for the hiring team or candidate to bridge these gaps during onboarding.
        """
        response = self._call_llm(prompt)
        if response:
            return response.strip()

        return (
            f"### Skill Gap Analysis for {candidate_name}\n"
            f"The candidate is missing key skills: **{', '.join(missing_skills)}**.\n\n"
            f"**Actionable Recommendations:**\n"
            f"1. **Targeted Onboarding**: Provide focused training modules in {missing_skills[0]}.\n"
            f"2. **Pair Programming**: Pair candidate with senior engineers for hands-on exposure to missing tools.\n"
            f"3. **Milestone Review**: Set a 30-day technical check-in focusing on domain mastery."
        )
