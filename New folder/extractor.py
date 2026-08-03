"""
Extractor Module for AI Resume Screening Agent.
Performs structured information extraction from Resumes and Job Descriptions.
Uses a hybrid architecture combining LLM structured parsing (Groq/OpenAI) with fallback rule-based regex/NLP extraction.
"""

import re
import json
import logging
from typing import Dict, Any, List
from config import GROQ_API_KEY, OPENAI_API_KEY, MODEL_PROVIDER, GROQ_MODEL, OPENAI_MODEL, COMMON_SKILLS_TAXONOMY

logger = logging.getLogger(__name__)

# Try importing LLM clients safely
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


class Extractor:
    """Hybrid Extractor combining LLM JSON parsing with Rule-Based Regex fallbacks."""

    def __init__(self, api_key: str = None, provider: str = None):
        self.api_key = api_key or GROQ_API_KEY or OPENAI_API_KEY
        self.provider = provider or MODEL_PROVIDER
        
        self.groq_client = None
        self.openai_client = None

        if self.provider == "groq" and GROQ_AVAILABLE and self.api_key:
            try:
                self.groq_client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

        if self.provider == "openai" and OPENAI_AVAILABLE and self.api_key:
            try:
                self.openai_client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

    # ==========================================
    # RULE-BASED REGEX & TAXONOMY FALLBACKS
    # ==========================================

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
            # Skip lines with contact info or long sentences
            if not re.search(r'(@|http|www|\d{5,})', line) and len(line.split()) <= 4 and len(line) < 40:
                # Remove titles if present
                clean = re.sub(r'^(resume|cv|curriculum vitae|profile)\s*[-:]?\s*', '', line, flags=re.IGNORECASE)
                if clean:
                    return clean.title()

        # Fallback to filename
        if filename:
            name_part = re.sub(r'[\-_]', ' ', filename.rsplit('.', 1)[0])
            name_part = re.sub(r'\b(resume|cv|profile|doc|pdf)\b', '', name_part, flags=re.IGNORECASE).strip()
            if name_part:
                return name_part.title()

        return "Candidate"

    @staticmethod
    def extract_skills_rule_based(text: str) -> List[str]:
        """Extract skills matching known taxonomy terms."""
        text_lower = text.lower()
        found_skills = set()

        for skill in COMMON_SKILLS_TAXONOMY:
            # Word boundary search for precision
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)

        return sorted(list(found_skills))

    @staticmethod
    def extract_experience_years_rule_based(text: str) -> float:
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
        
        # Estimate from date ranges e.g. 2018 - 2023
        date_ranges = re.findall(r'\b(20\d{2}|19\d{2})\s*[-–\text{to}]\s*(20\d{2}|present|current)\b', text_lower)
        total_months = 0
        for start, end in date_ranges:
            start_yr = int(start)
            end_yr = 2026 if end in ['present', 'current'] else int(end)
            if end_yr >= start_yr:
                total_months += (end_yr - start_yr) * 12
        
        if total_months > 0:
            return round(total_months / 12.0, 1)

        return 2.0  # Default neutral baseline if unstated

    @staticmethod
    def extract_education_rule_based(text: str) -> List[str]:
        """Extract degree and educational background using keyword heuristics."""
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

    # ==========================================
    # LLM POWERED STRUCTURED EXTRACTION
    # ==========================================

    def _call_llm(self, prompt: str) -> str:
        """Helper to invoke Groq or OpenAI LLM."""
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq LLM call error: {e}")

        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI LLM call error: {e}")

        return ""

    def extract_resume_info(self, resume_text: str, filename: str = "") -> Dict[str, Any]:
        """
        Extract structured resume data (Name, Email, Phone, Skills, Education, Experience, Certifications, Projects).
        Tries LLM extraction first, falling back to rule-based NLP parser.
        """
        if not resume_text or len(resume_text.strip()) < 20:
            return self._default_resume_dict(filename)

        prompt = f"""
        Analyze the following candidate resume text and extract structured JSON data.
        
        JSON schema required:
        {{
            "name": "Candidate Full Name",
            "email": "Email address or Not Found",
            "phone": "Phone number or Not Found",
            "skills": ["Skill1", "Skill2", ...],
            "experience_years": 5.0,
            "education": ["Degree Name, Field of Study, Institution"],
            "certifications": ["Certification 1", ...],
            "projects": ["Project Title/Description", ...]
        }}

        Resume Text:
        \"\"\"{resume_text[:4000]}\"\"\"
        """

        llm_response = self._call_llm(prompt)
        if llm_response:
            try:
                data = json.loads(llm_response)
                # Validation & Fallback injection
                if not data.get("name") or data.get("name") == "Candidate Full Name":
                    data["name"] = self.extract_name(resume_text, filename)
                if not data.get("email") or data.get("email") == "Not Found":
                    data["email"] = self.extract_email(resume_text)
                if not data.get("phone") or data.get("phone") == "Not Found":
                    data["phone"] = self.extract_phone(resume_text)
                if not data.get("skills"):
                    data["skills"] = self.extract_skills_rule_based(resume_text)
                if "experience_years" not in data or not isinstance(data["experience_years"], (int, float)):
                    data["experience_years"] = self.extract_experience_years_rule_based(resume_text)
                if not data.get("education"):
                    data["education"] = self.extract_education_rule_based(resume_text)
                if "certifications" not in data:
                    data["certifications"] = []
                if "projects" not in data:
                    data["projects"] = []

                data["raw_text"] = resume_text
                return data
            except Exception as e:
                logger.warning(f"Failed to parse LLM JSON response: {e}")

        # Rule-based fallback
        return {
            "name": self.extract_name(resume_text, filename),
            "email": self.extract_email(resume_text),
            "phone": self.extract_phone(resume_text),
            "skills": self.extract_skills_rule_based(resume_text),
            "experience_years": self.extract_experience_years_rule_based(resume_text),
            "education": self.extract_education_rule_based(resume_text),
            "certifications": [],
            "projects": [],
            "raw_text": resume_text
        }

    def extract_job_description_info(self, jd_text: str) -> Dict[str, Any]:
        """
        Extract structured Job Description requirements:
        Required Skills, Min Experience Required, Education Required, Responsibilities, Preferred Skills.
        """
        if not jd_text or len(jd_text.strip()) < 20:
            return self._default_jd_dict()

        prompt = f"""
        Analyze the following Job Description (JD) text and extract structured JSON data.

        JSON schema required:
        {{
            "job_title": "Target Role Title",
            "required_skills": ["Skill1", "Skill2", ...],
            "preferred_skills": ["SkillA", "SkillB", ...],
            "experience_required_years": 3.0,
            "education_required": "Bachelor's degree in CS or related field",
            "responsibilities": ["Responsibility 1", ...]
        }}

        Job Description Text:
        \"\"\"{jd_text[:4000]}\"\"\"
        """

        llm_response = self._call_llm(prompt)
        if llm_response:
            try:
                data = json.loads(llm_response)
                if not data.get("required_skills"):
                    data["required_skills"] = self.extract_skills_rule_based(jd_text)
                if "experience_required_years" not in data or not isinstance(data["experience_required_years"], (int, float)):
                    data["experience_required_years"] = self.extract_experience_years_rule_based(jd_text)
                if not data.get("education_required"):
                    data["education_required"] = "Bachelor's Degree"
                if "preferred_skills" not in data:
                    data["preferred_skills"] = []
                if "responsibilities" not in data:
                    data["responsibilities"] = []

                data["raw_text"] = jd_text
                return data
            except Exception as e:
                logger.warning(f"Failed to parse JD LLM response: {e}")

        # Rule-based fallback
        skills = self.extract_skills_rule_based(jd_text)
        return {
            "job_title": "Job Position",
            "required_skills": skills[:7] if skills else ["Python", "Problem Solving"],
            "preferred_skills": skills[7:] if len(skills) > 7 else [],
            "experience_required_years": self.extract_experience_years_rule_based(jd_text),
            "education_required": "Bachelor's Degree in Computer Science or related field",
            "responsibilities": [],
            "raw_text": jd_text
        }

    def _default_resume_dict(self, filename: str) -> Dict[str, Any]:
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

    def _default_jd_dict(self) -> Dict[str, Any]:
        return {
            "job_title": "Position",
            "required_skills": [],
            "preferred_skills": [],
            "experience_required_years": 0.0,
            "education_required": "Bachelor's Degree",
            "responsibilities": [],
            "raw_text": ""
        }
