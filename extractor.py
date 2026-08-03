"""
Extractor Module for AI Resume Screening Agent.
Performs structured information extraction from Resumes and Job Descriptions.
Uses a fast, rule-based NLP taxonomy engine for 100% offline, self-contained processing.
"""

import re
import logging
from typing import Dict, Any, List
from config import COMMON_SKILLS_TAXONOMY

logger = logging.getLogger(__name__)


class Extractor:
    """Fast NLP Extractor for candidate resumes and Job Descriptions."""

    @staticmethod
    def extract_email(text: str) -> str:
        """Extract email address using regex."""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(email_pattern, text)
        return match.group(0) if match else "Not Found"

    @staticmethod
    def extract_phone(text: str) -> str:
        """Extract phone number using regex."""
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
        match = re.search(phone_pattern, text)
        return match.group(0) if match else "Not Found"

    @staticmethod
    def extract_name(text: str, filename: str = "") -> str:
        """Extract candidate name from header or filename."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in lines[:5]:
            if not re.search(r'(@|http|www|\d{5,})', line) and len(line.split()) <= 4 and len(line) < 40:
                clean = re.sub(r'^(resume|cv|curriculum vitae|profile)\s*[-:]?\s*', '', line, flags=re.IGNORECASE)
                if clean:
                    return clean.title()

        if filename:
            name_part = re.sub(r'[\-_]', ' ', filename.rsplit('.', 1)[0])
            name_part = re.sub(r'\b(resume|cv|profile|doc|pdf)\b', '', name_part, flags=re.IGNORECASE).strip()
            if name_part:
                return name_part.title()

        return "Candidate"

    @staticmethod
    def extract_skills(text: str) -> List[str]:
        """Extract skills matching known taxonomy terms."""
        text_lower = text.lower()
        found_skills = set()

        for skill in COMMON_SKILLS_TAXONOMY:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)

        return sorted(list(found_skills))

    @staticmethod
    def extract_experience_years(text: str) -> float:
        """Extract total years of experience using regex patterns."""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)',
            r'(?:experience|exp)\s*:\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)',
            r'(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:in|of)?\s*(?:software|data|engineering|development|work)',
        ]
        text_lower = text.lower()
        years = []
        for p in patterns:
            matches = re.findall(p, text_lower)
            for m in matches:
                try:
                    years.append(float(m))
                except ValueError:
                    pass
        if years:
            return max(years)
        
        date_ranges = re.findall(r'\b(20\d{2}|19\d{2})\s*[-–to]\s*(20\d{2}|present|current)\b', text_lower)
        total_months = 0
        for start, end in date_ranges:
            start_yr = int(start)
            end_yr = 2026 if end in ['present', 'current'] else int(end)
            if end_yr >= start_yr:
                total_months += (end_yr - start_yr) * 12
        
        if total_months > 0:
            return round(total_months / 12.0, 1)

        return 2.0

    @staticmethod
    def extract_education(text: str) -> List[str]:
        """Extract educational background degrees."""
        degrees = []
        patterns = [
            r'\b(ph\.?d|doctorate)\b',
            r'\b(master[\'s]*|m\.s\.?|m\.tech|m\.e\.?|mba)\b',
            r'\b(bachelor[\'s]*|b\.s\.?|b\.tech|b\.e\.?|b\.a\.?)\b',
            r'\b(associate|diploma)\b',
        ]
        text_lower = text.lower()
        for p in patterns:
            matches = re.findall(p, text_lower)
            if matches:
                degrees.extend(matches)
        
        return list(set([d.upper() for d in degrees])) if degrees else ["Bachelor's Degree"]

    def extract_resume_info(self, resume_text: str, filename: str = "") -> Dict[str, Any]:
        """Extract structured resume data (Name, Email, Phone, Skills, Education, Experience)."""
        if not resume_text or len(resume_text.strip()) < 10:
            return {
                "name": self.extract_name("", filename),
                "email": "Not Found",
                "phone": "Not Found",
                "skills": [],
                "experience_years": 0.0,
                "education": ["Not Found"],
                "certifications": [],
                "projects": [],
                "raw_text": ""
            }

        return {
            "name": self.extract_name(resume_text, filename),
            "email": self.extract_email(resume_text),
            "phone": self.extract_phone(resume_text),
            "skills": self.extract_skills(resume_text),
            "experience_years": self.extract_experience_years(resume_text),
            "education": self.extract_education(resume_text),
            "certifications": [],
            "projects": [],
            "raw_text": resume_text
        }

    def extract_job_description_info(self, jd_text: str) -> Dict[str, Any]:
        """Extract structured Job Description details (Required Skills, Experience, Education)."""
        if not jd_text or len(jd_text.strip()) < 10:
            return {
                "job_title": "Position",
                "required_skills": [],
                "preferred_skills": [],
                "experience_required_years": 0.0,
                "education_required": "Bachelor's Degree",
                "responsibilities": [],
                "raw_text": ""
            }

        skills = self.extract_skills(jd_text)
        return {
            "job_title": "Job Position",
            "required_skills": skills[:7] if skills else ["Python", "Problem Solving"],
            "preferred_skills": skills[7:] if len(skills) > 7 else [],
            "experience_required_years": self.extract_experience_years(jd_text),
            "education_required": "Bachelor's Degree in CS or related field",
            "responsibilities": [],
            "raw_text": jd_text
        }
