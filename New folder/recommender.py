"""
Recommender Module for AI Resume Screening Agent.
Generates candidate recommendations, qualitative evaluation reports, strengths/weaknesses,
score explanations, tailored interview questions, and skill gap analyses 100% offline.
"""

import logging
from typing import Dict, Any, List
from config import RECOMMENDATION_THRESHOLDS

logger = logging.getLogger(__name__)


class CandidateRecommender:
    """Generates candidate recommendations, qualitative evaluation reports, and interview kits."""

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

    @staticmethod
    def generate_interview_questions(candidate_name: str, jd_title: str, matched_skills: List[str], missing_skills: List[str]) -> List[str]:
        """Generates 5 tailored technical and behavioral interview questions based on profile overlap."""
        q1_skill = matched_skills[0] if matched_skills else "your core technical stack"
        q2_skill = missing_skills[0] if missing_skills else "a tool you are learning"

        return [
            f"1. **Core Competency**: Walk us through a recent production application where you utilized **{q1_skill}**.",
            f"2. **Technical Gap Handling**: How would you ramp up and overcome challenges if assigned a task involving **{q2_skill}**?",
            "3. **Architecture & Design**: Can you describe how you balance code maintainability with tight project deadlines?",
            "4. **Problem Solving**: Tell us about the most complex technical bug or system bottleneck you solved recently.",
            "5. **Teamwork & Collaboration**: How do you approach code reviews and architectural discussions with team members?"
        ]

    @staticmethod
    def explain_score(candidate_name: str, score_eval: Dict[str, Any]) -> str:
        """Generates an executive natural-language explanation of how the candidate's score was derived."""
        sub = score_eval["sub_scores"]
        overall = score_eval["overall_score"]
        
        return (
            f"Candidate **{candidate_name}** achieved an overall score of **{overall}/100** based on our weighted evaluation model.\n\n"
            f"- **Skill Match (40% weight)**: {sub['skill_score']}% match based on taxonomy coverage.\n"
            f"- **Experience Match (30% weight)**: {sub['experience_score']}% score based on years in industry.\n"
            f"- **Education Match (15% weight)**: {sub['education_score']}% level alignment.\n"
            f"- **Semantic Similarity (15% weight)**: {sub['semantic_score']}% vector similarity."
        )

    @staticmethod
    def analyze_skill_gap(candidate_name: str, missing_skills: List[str], required_skills: List[str]) -> str:
        """Generates actionable skill gap analysis and onboarding recommendations."""
        if not missing_skills:
            return f"🎉 **{candidate_name}** possesses all required skills for this position!"

        return (
            f"### 🎯 Skill Gap Analysis for {candidate_name}\n\n"
            f"**Missing Required Skills**: `{', '.join(missing_skills)}`\n\n"
            f"**Recommended Action Plan:**\n"
            f"1. **Targeted Onboarding**: Assign focused tutorials in `{missing_skills[0]}` during week 1.\n"
            f"2. **Peer Mentorship**: Pair candidate with a senior developer experienced in `{missing_skills[-1]}`.\n"
            f"3. **Practical Milestone**: Schedule a 30-day technical check-in to verify domain mastery."
        )
