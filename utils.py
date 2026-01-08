import os
import re
import json
import time
import requests
import pandas as pd

import google.generativeai as genai
import asyncio

# --- SCHEMAS ---

RUBRIC_EXTRACTION_SCHEMA = {
    "type": "object",
    "description": "Complete grading rubric with step-wise marking breakdown",
    "properties": {
        "exam_metadata": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Subject name"},
                "grade": {"type": "string", "description": "Grade level"},
                "exam_name": {"type": "string"},
                "total_marks": {"type": "string"},
                "total_questions": {"type": "string"},
                "duration": {"type": "string"},
                "instructions": {"type": "string"}
            }
        },
        "section_info": {
            "type": "array",
            "description": "Section-wise metadata",
            "items": {
                "type": "object",
                "properties": {
                    "section_name": {"type": "string"},
                    "question_range": {"type": "string"},
                    "answer_requirement": {"type": "string"},
                    "marks_per_question": {"type": "string"},
                    "answer_length_limit": {"type": "string"}
                }
            }
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_no": {"type": "string"},
                    "section": {"type": "string"},
                    "question_type": {"type": "string"},
                    "difficulty_level": {"type": "string"},
                    "question_text_plain": {"type": "string"},
                    "question_math_latex": {"type": "string", "description": "Question text with math formatted in LaTeX, wrapped in $ symbols (e.g. $x^2$)"},
                    "figure_summary_rubric": {"type": "string"},
                    "correct_answer_plain": {"type": "string"},
                    "correct_answer_latex": {"type": "string", "description": "Answer with math formatted in LaTeX, wrapped in $ symbols"},
                    "max_marks": {"type": "string"},
                    "marking_scheme": {
                        "type": "string",
                        "description": "Free-text marking guide (kept for reference)"
                    },
                    "step_marking": {
                        "type": "array",
                        "description": "Step-wise marking rubric. Each element represents a logical concept/step.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "marksplit": {"type": "number", "description": "Marks allocated to this concept/step"},
                                "step_wise_answer": {"type": "string", "description": "Concept description. Use $...$ for all match expressions (e.g. $x=5$)."},
                                "diagram_description": {"type": "string", "description": "Optional: diagram/label requirement for this step"}
                            }
                        }
                    },
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "Key concepts to check"},
                    "diagram_labeling_requirements": {"type": "string"}
                }
            }
        }
    }
}

STUDENT_EXTRACTION_SCHEMA = {
    "type": "object",
    "description": "Student exam answers with complete math, figure, and metadata support",
    "properties": {
        "student_metadata": {
            "type": "object",
            "properties": {
                "student_name": {"type": "string"},
                "roll_number": {"type": "string"},
                "class_section": {"type": "string"},
                "exam_date": {"type": "string"}
            }
        },
        "answers": {
            "type": "array",
            "description": "Complete student answers with text & math (exclude crossed-out work)",
            "items": {
                "type": "object",
                "properties": {
                    "question_no": {"type": "string", "description": "Normalized question number"},
                    "page_number": {"type": "string", "description": "Page where answer appears"},
                    "answer_sequence_position": {"type": "string", "description": "Position in student's writing order"},
                    "section_group": {"type": "string", "description": "Detected section grouping"},
                    "answer_text_plain": {
                        "type": "string",
                        "description": "Student answer. WRAP ALL MATH EXPRESSIONS IN $...$ delimiters for LaTeX rendering."
                    },
                    "figure_summary_student": {"type": "string", "description": "Textual description of student-drawn diagram"},
                    "status": {"type": "string", "description": "Attempted, Blank, or Partial"}
                }
            }
        }
    }
}

# --- HELPER FUNCTIONS ---

def normalize_qno(qno: str) -> str:
    """Normalize question numbers: Q1., 1., 1) -> 1"""
    if not qno:
        return ""
    q = str(qno).strip()
    q = q.lstrip("Qq").rstrip(".").strip()
    return q

def safe_get_string(obj, key, default=""):
    """Safely get string/list from dict"""
    if not obj:
        return default
    value = obj.get(key, default)
    if value is None:
        return default
    if isinstance(value, (str, list)):
        return value
    return str(value)

def normalize_step_marking(reference_rubric):
    """Normalize step_marking so that sum of marksplit equals max_marks"""
    if not reference_rubric:
        return reference_rubric

    for q in reference_rubric.get("questions", []):
        try:
            max_marks = float(str(q.get("max_marks", "0")))
        except:
            max_marks = 0.0

        steps = q.get("step_marking") or []
        if not steps or max_marks <= 0:
            continue

        total_step_marks = 0.0
        for step in steps:
            try:
                ms = float(step.get("marksplit", 0))
            except:
                ms = 0.0
            step["marksplit"] = ms
            total_step_marks += ms

        if total_step_marks <= 0:
            continue

        scale = max_marks / total_step_marks
        for step in steps:
            step["marksplit"] = round(step["marksplit"] * scale, 2)
            
    return reference_rubric

# --- API CLIENTS ---

def call_marker_with_structured_extraction(filepath, api_key, page_schema, max_retries=3):
    """Call Datalab Marker API"""
    DATALAB_MARKER_ENDPOINT = "https://www.datalab.to/api/v1/marker"
    
    for attempt in range(max_retries):
        try:
            with open(filepath, 'rb') as f:
                form_data = {
                    'file': (os.path.basename(filepath), f, 'application/pdf'),
                    'page_schema': (None, json.dumps(page_schema)),
                    'output_format': (None, 'json'),
                    'use_llm': (None, 'true'),
                    'force_ocr': (None, 'true'),
                }
                headers = {'X-Api-Key': api_key}
                response = requests.post(DATALAB_MARKER_ENDPOINT, files=form_data, headers=headers)
                data = response.json()

                if not data.get('success'):
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue

                check_url = data.get('request_check_url')
                
                # Poll for completion
                for i in range(600):
                    time.sleep(2)
                    resp = requests.get(check_url, headers=headers)
                    result = resp.json()
                    
                    if result.get('status') == 'complete':
                        return result, None
                    elif result.get('status') == 'error':
                        return None, f"Extraction Error: {result.get('error', 'Unknown error')}"
                    
                return None, "Timeout: Extraction took too long." # Timeout

        except Exception as e:
            if attempt < max_retries - 1:
                 time.sleep(2 ** attempt)
            else:
                return None, f"Request Failed: {str(e)}"
    return None, "Max retries exceeded or API unreachable."

def extract_structured_json(marker_result):
    """Extract and parse JSON from Marker result"""
    if not marker_result or not marker_result.get('success'):
        return None, None

    extraction_json_str = marker_result.get('extraction_schema_json')
    if not extraction_json_str:
        return None, None

    try:
        extracted_data = json.loads(extraction_json_str)
        citations = marker_result.get('json')
        return extracted_data, citations
    except:
        return None, None

def extract_json_robust(text):
    """Clean JSON from LLM response"""
    text = text.strip()
    json_match = re.search(r'```json?\s*\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
        
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        return text[start:end+1]
    return text.strip()

def parse_json_fallbacks(json_str):
    """Parse JSON with fallbacks, specifically handling LaTeX backslashes"""
    # 1. Try standard clean
    json_clean = re.sub(r'\\n|\\t', ' ', json_str)
    json_clean = re.sub(r'\s+', ' ', json_clean).strip()
    
    try:
        return json.loads(json_clean)
    except:
        pass

    # 2. Try raw string as sometimes regex hurts more than helps
    try:
        return json.loads(json_str)
    except:
        pass

    # 3. Try to fix single backslashes for LaTeX (e.g. \frac -> \\frac)
    # This looks for backslashes that are NOT followed by specific JSON control chars
    # We want to target things like \t, \f, \r, \n which ARE valid JSON escapes if intended as control chars
    # But often LLM outputs \frac which is invalid.
    # A simple heuristic: Replace single \ with \\, but respect \\ already existing.
    
    # Negative lookbehind to ensure we don't start with a backslash (i.e. strictly odd number of backslashes)
    # actually, simpler approach: use the raw string and re-escape.
    
    try:
        # crude fix: replace all single backslashes with double, 
        # BUT this ruins valid escapes like \" or \n if we aren't careful.
        # Let's try to specifically target common LaTeX patterns if simple parse fails.
        # This is hard to do perfectly with regex.
        # Alternative: use a more lenient parser or just try double escaping everything that looks like a latex command
        
        # Heuristic: Replace \ with \\ if it's followed by a letter
        fixed_str = re.sub(r'\\([a-zA-Z])', r'\\\\\1', json_str)
        return json.loads(fixed_str)
    except:
        pass

    return None

def call_gemini_with_retries(model, eval_prompt, question_ref, max_retries=3):
    """Call Gemini API for evaluation"""
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                eval_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.05,
                    max_output_tokens=1024
                )
            )
            json_str = extract_json_robust(response.text)
            result = parse_json_fallbacks(json_str)
            if result:
                return result
        except Exception as e:
            delay = base_delay * (2 ** attempt)
            if "429" in str(e):
                delay *= 2
            time.sleep(delay)
            # Capture last error
            if attempt == max_retries - 1:
                return {
                    "question_no": safe_get_string(question_ref, "question_no"),
                    "marks_awarded": "ERROR",
                    "max_marks": safe_get_string(question_ref, "max_marks"),
                    "feedback": f"Failed to evaluate. Error: {str(e)}",
                    "status": "Error",
                    "error_details": f"Error: {str(e)}\n\nRaw Response:\n{response.text if hasattr(response, 'text') else 'No response text'}"
                }

    return {
        "question_no": safe_get_string(question_ref, "question_no"),
        "marks_awarded": "ERROR",
        "max_marks": safe_get_string(question_ref, "max_marks"),
        "feedback": "Failed to evaluate (Max Retries)",
        "status": "Error",
        "error_details": "Max retries exceeded during Gemini call."
    }

# --- EVALUATION LOGIC ---

def evaluate_single_answer(model, question_ref, student_answer_text, student_status, student_figures=""):
    """Evaluate a single answer using Gemini"""
    fig_clean = (student_figures or "").strip()
    if not (student_answer_text or "").strip() and not fig_clean:
        return {
            "question_no": safe_get_string(question_ref, "question_no"),
            "marks_awarded": "0",
            "max_marks": safe_get_string(question_ref, "max_marks", "0"),
            "feedback": "Answer not attempted",
            "stepwise_feedback": [],
            "diagram_feedback": "N/A",
            "status": "Blank"
        }

    eval_prompt = f"""You are an STEM subject expert examiner grading subjective & objective student answer for questions using detailed rubric.

==QUESTION DETAILS==
Question No: {safe_get_string(question_ref, 'question_no')}
Question Type: {safe_get_string(question_ref, 'question_type')}
Question Text: {safe_get_string(question_ref, 'question_text_plain')}

==RUBRIC REQUIREMENTS==
Max Marks: {safe_get_string(question_ref, 'max_marks', '5')}
Marking Scheme (STEPWISE):
{safe_get_string(question_ref, 'marking_scheme', 'Grade on correctness, completeness, and reasoning')}

Keywords to Check: {', '.join(safe_get_string(question_ref, 'keywords', []))}

==REFERENCE SOLUTION==
Plain Text Answer:
{safe_get_string(question_ref, 'correct_answer_plain')}

==STUDENT'S ANSWER==
Plain Text:
{student_answer_text}

Diagrams/Figures:
{student_figures}

==EVALUATION REQUIREMENTS==
1. Check for conceptual correctness.
2. Award partial credit for correct steps even if final answer is wrong.
3. Ignore minor spelling mistakes.

==OUTPUT FORMAT==
Provide evaluation in this EXACT JSON format:
{{
  "question_no": "{safe_get_string(question_ref, 'question_no')}",
  "marks_awarded": <number>,
  "max_marks": "{safe_get_string(question_ref, 'max_marks')}",
  "stepwise_feedback": [
    {{
      "step_id": 1,
      "description": "<concept>",
      "marks_awarded": <number>,
      "max_marks": <number>,
      "feedback": "<short feedback. IMPORTANT: Use double backslashes for ALL LaTeX math, e.g. \\\\frac, \\\\tanh.>"
    }}
  ],
  "overall_feedback": "<summary>",
  "status": "Attempted" | "Correct" | "Partial" | "Incorrect"
}}"""

    return call_gemini_with_retries(model, eval_prompt, question_ref)

def postprocess_evaluation(eval_result, max_marks):
    """Normalize status based on marks"""
    if not eval_result:
        return {"status": "Error"}
    
    marks_str = str(eval_result.get("marks_awarded", "0")).strip()
    if marks_str == "ERROR":
        return eval_result
        
    try:
        marks = float(marks_str)
        max_m = float(max_marks or 0)
        if max_m > 0 and marks >= max_m * 0.9:
            eval_result["status"] = "Correct"
        elif marks > 0:
            eval_result["status"] = "Attempted"
        else:
            eval_result["status"] = "Blank"
    except:
        eval_result["status"] = "Error"
    return eval_result
