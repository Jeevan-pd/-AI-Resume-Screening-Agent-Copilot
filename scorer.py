"""
Scorer Module for AI Resume Screening Agent.
Calculates candidate match sub-scores and overall score based on the 40/30/15/15 weighting formula.
"""

import logging
from typing import Dict, Any, List, Tuple
from config import WEIGHTS, EDUCATION_HIERARCHY

logger = logging.getLogger(__name__)


class CandidateScorer:
    """Scoring engine evaluating candidates against job requirements."""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or WEIGHTS

    @staticmethod
    def compute_skill_match(candidate_skills: List[str], required_skills: List[str], preferred_skills: List[str] = None) -> Tuple[float, List[str], List[str]]:
        """
        Computes Skill Match % based on overlap with required & preferred skills.
        Returns (skill_score, matching_skills, missing_skills).
        """
        if not required_skills:
            return 100.0, candidate_skills, []

        cand_set = set([s.lower().strip() for s in candidate_skills])
        req_set = set([s.lower().strip() for s in required_skills])
        pref_set = set([s.lower().strip() for s in (preferred_skills or [])])

        # Matched and missing required skills
        matched_required = req_set.intersection(cand_set)
        missing_required = req_set - cand_set

        # Matched preferred skills
        matched_preferred = pref_set.intersection(cand_set)

        # Base mandatory coverage (80% of total skill weight)
        mandatory_score = (len(matched_required) / len(req_set)) * 80.0 if req_set else 80.0

        # Preferred bonus coverage (20% of total skill weight)
        preferred_score = (len(matched_preferred) / len(pref_set)) * 20.0 if pref_set else 20.0

        total_skill_score = min(100.0, mandatory_score + preferred_score)

        # Format original casing for output display
        matched_display = [s for s in candidate_skills if s.lower().strip() in cand_set.intersection(req_set.union(pref_set))]
        missing_display = [s for s in required_skills if s.lower().strip() in missing_required]

        return round(total_skill_score, 2), matched_display, missing_display

    @staticmethod
    def compute_experience_match(candidate_exp_years: float, required_exp_years: float) -> float:
        """
        Computes Experience Match %.
        - If candidate meets or exceeds required years: 100.0
        - If required is 0: 100.0
        - If less: proportional percentage with minimum 30% credit for any entry-level experience.
        """
        if required_exp_years <= 0:
            return 100.0

        if candidate_exp_years >= required_exp_years:
            # Full score with up to 10% bonus for surplus experience (capped at 100)
            bonus = min(10.0, (candidate_exp_years - required_exp_years) * 2.0)
            return min(100.0, round(90.0 + bonus, 2))

        # Proportional score
        ratio = candidate_exp_years / required_exp_years
        score = max(30.0, ratio * 90.0)
        return round(score, 2)

    @staticmethod
    def compute_education_match(candidate_education: List[str], required_education: str) -> float:
        """
        Computes Education Match % by comparing highest qualification level in hierarchy.
        """
        if not required_education or required_education.lower() == "not required":
            return 100.0

        # Determine target requirement level
        target_level = 3  # Default Bachelor's level
        req_lower = required_education.lower()
        for edu_key, level in EDUCATION_HIERARCHY.items():
            if edu_key in req_lower:
                target_level = max(target_level, level)
                break

        # Determine candidate highest level
        candidate_max_level = 1
        cand_text = " ".join(candidate_education).lower()
        for edu_key, level in EDUCATION_HIERARCHY.items():
            if edu_key in cand_text:
                candidate_max_level = max(candidate_max_level, level)

        if candidate_max_level >= target_level:
            return 100.0
        elif candidate_max_level == target_level - 1:
            return 75.0
        else:
            return 50.0

    def evaluate_candidate(
        self,
        candidate_info: Dict[str, Any],
        jd_info: Dict[str, Any],
        semantic_score: float
    ) -> Dict[str, Any]:
        """
        Evaluates a single candidate and calculates sub-scores and overall score.
        Formula: 40% Skill Match + 30% Experience + 15% Education + 15% Semantic Similarity
        """
        # 1. Skill Match
        skill_score, matched_skills, missing_skills = self.compute_skill_match(
            candidate_skills=candidate_info.get("skills", []),
            required_skills=jd_info.get("required_skills", []),
            preferred_skills=jd_info.get("preferred_skills", [])
        )

        # 2. Experience Match
        exp_score = self.compute_experience_match(
            candidate_exp_years=candidate_info.get("experience_years", 0.0),
            required_exp_years=jd_info.get("experience_required_years", 0.0)
        )

        # 3. Education Match
        edu_score = self.compute_education_match(
            candidate_education=candidate_info.get("education", []),
            required_education=jd_info.get("education_required", "Bachelor's Degree")
        )

        # 4. Overall Score calculation using formula
        w = self.weights
        overall_score = (
            (w["skill_match"] * skill_score) +
            (w["experience_match"] * exp_score) +
            (w["education_match"] * edu_score) +
            (w["semantic_similarity"] * semantic_score)
        )
        overall_score = round(min(100.0, max(0.0, overall_score)), 2)

        return {
            "overall_score": overall_score,
            "sub_scores": {
                "skill_score": round(skill_score, 2),
                "experience_score": round(exp_score, 2),
                "education_score": round(edu_score, 2),
                "semantic_score": round(semantic_score, 2),
            },
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
        }
