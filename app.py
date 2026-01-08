import streamlit as st
import pandas as pd
import os
import json
import time
import tempfile
import google.generativeai as genai
from utils import (
    RUBRIC_EXTRACTION_SCHEMA,
    STUDENT_EXTRACTION_SCHEMA,
    normalize_step_marking,
    call_marker_with_structured_extraction,
    extract_structured_json,
    evaluate_single_answer,
    postprocess_evaluation,
    normalize_qno,
    safe_get_string
)

st.set_page_config(page_title="Auto-Grader AI", layout="wide")

# --- STATE MANAGEMENT ---
if "rubric_data" not in st.session_state:
    st.session_state.rubric_data = None
if "student_data_list" not in st.session_state:
    st.session_state.student_data_list = []
if "grading_results" not in st.session_state:
    st.session_state.grading_results = []

# --- CONFIGURATION ---
# Helper to get secrets checking both st.secrets and os.environ
def get_secret(key):
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, "")

datalab_key = get_secret("DATALAB_API_KEY")
gemini_key = get_secret("GEMINI_API_KEY")

if not datalab_key or not gemini_key:
    st.warning("⚠️ One or more API keys are missing. Please set DATALAB_API_KEY and GEMINI_API_KEY in secrets or environment variables.")

if gemini_key:
    genai.configure(api_key=gemini_key)
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
    except:
        st.error("Invalid Gemini Key in secrets")
        model = None
else:
    model = None
    st.warning("GEMINI_API_KEY not found in secrets.")

if not datalab_key:
    st.warning("DATALAB_API_KEY not found in secrets.")
    
# --- MAIN APP ---
st.title("📝 AI Exam Auto-Grader")
st.markdown("Upload a solution rubric and student answer scripts to automatically grade them.")

tab1, tab2, tab3 = st.tabs(["1. Upload Rubric", "2. Upload Students", "3. Grading & Results"])

# --- TAB 1: RUBRIC ---
with tab1:
    st.header("Step 1: Upload Reference Solution (Rubric)")
    uploaded_rubric = st.file_uploader("Upload Solution PDF", type=["pdf"])
    
    if uploaded_rubric and datalab_key:
        if st.button("Extract Rubric"):
            with st.spinner("Analyzing Rubric PDF..."):
                # Save to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_rubric.getbuffer())
                    rubric_path = tmp.name
                
                # Call API
                result, error_msg = call_marker_with_structured_extraction(
                    rubric_path, datalab_key, RUBRIC_EXTRACTION_SCHEMA
                )
                
                if result:
                    rubric, _ = extract_structured_json(result)
                    if rubric:
                        rubric = normalize_step_marking(rubric)
                        st.session_state.rubric_data = rubric
                        st.success("✅ Rubric Extracted Successfully!")
                        st.json(rubric.get("exam_metadata", {}))
                        
                        # Show Schema
                        questions = rubric.get("questions", [])
                        st.info(f"Found {len(questions)} questions in rubric.")

                        with st.expander("View Extracted Rubric Details"):
                            for q in questions:
                                st.markdown(f"**Q{q.get('question_no')} ({q.get('max_marks')} marks)**")
                                # Prefer LaTeX if available, else plain text
                                q_text = q.get("question_math_latex") or q.get("question_text_plain")
                                st.markdown(q_text)
                                
                                st.caption("Reference Answer:")
                                ans_text = q.get("correct_answer_latex") or q.get("correct_answer_plain")
                                st.markdown(ans_text)

                                if q.get("step_marking"):
                                    st.caption("Step-wise Marking Scheme:")
                                    for step in q.get("step_marking"):
                                        marks = step.get("marksplit", 0)
                                        desc = step.get("step_wise_answer", "")
                                        if desc:
                                            st.markdown(f"• **[{marks} Marks]** {desc}")
                                st.divider()
                    else:
                        st.error("Failed to parse rubric JSON.")
                else:
                    st.error(f"Rubric extraction failed: {error_msg}")
                
                # Cleanup
                if os.path.exists(rubric_path):
                    os.remove(rubric_path)

    if st.session_state.rubric_data:
        st.success(f"Active Rubric: {st.session_state.rubric_data.get('exam_metadata', {}).get('exam_name', 'Unnamed')}")
        
        # Download Option
        rubric_json = json.dumps(st.session_state.rubric_data, indent=2)
        st.download_button(
            label="📥 Download Rubric JSON",
            data=rubric_json,
            file_name="rubric_extracted.json",
            mime="application/json"
        )

# --- TAB 2: STUDENTS ---
with tab2:
    st.header("Step 2: Upload Student Answer Scripts")
    uploaded_students = st.file_uploader("Upload Student PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_students and datalab_key and st.session_state.rubric_data:
        if st.button(f"Process {len(uploaded_students)} Student Files"):
            progress_bar = st.progress(0)
            st.session_state.student_data_list = [] # Reset on new process
            
            for idx, stu_file in enumerate(uploaded_students):
                status_text = st.empty()
                status_text.text(f"Processing {stu_file.name}...")
                
                # Save temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(stu_file.getbuffer())
                    stu_path = tmp.name
                
                # Call API
                result, error_msg = call_marker_with_structured_extraction(
                    stu_path, datalab_key, STUDENT_EXTRACTION_SCHEMA
                )
                
                if result:
                    data, _ = extract_structured_json(result)
                    if data:
                        data['filename'] = stu_file.name
                        st.session_state.student_data_list.append(data)
                        st.write(f"✅ Extracted: {data.get('student_metadata', {}).get('student_name', 'Unknown')}")
                        
                        with st.expander("View Extracted Student Answers"):
                            for ans in data.get("answers", []):
                                st.markdown(f"**Q{ans.get('question_no')}**")
                                st.markdown(ans.get("answer_text_plain", "No text extracted"))
                                if ans.get("figure_summary_student"):
                                    st.caption("Figure Summary:")
                                    st.info(ans.get("figure_summary_student"))
                                st.divider()
                    else:
                        st.warning(f"Failed to parse {stu_file.name}")
                else:
                    st.error(f"Failed to process {stu_file.name}: {error_msg}")
                
                # Cleanup
                if os.path.exists(stu_path):
                    os.remove(stu_path)
                
                progress_bar.progress((idx + 1) / len(uploaded_students))
                
            st.success(f"Processed {len(st.session_state.student_data_list)} students.")

    if st.session_state.student_data_list:
        st.info(f"{len(st.session_state.student_data_list)} students ready for grading.")
        
        # Download Option
        students_json = json.dumps(st.session_state.student_data_list, indent=2)
        st.download_button(
            label="📥 Download Students JSON",
            data=students_json,
            file_name="students_extracted.json",
            mime="application/json"
        )

# --- TAB 3: GRADING ---
with tab3:
    st.header("Step 3: AI Grading")
    
    if not st.session_state.rubric_data:
        st.warning("Please upload specific rubric first.")
    elif not st.session_state.student_data_list:
        st.warning("Please upload student scripts.")
    elif not model:
        st.error("Gemini API Key required.")
    else:
        if st.button("🚀 Start Grading"):
            all_evals = []
            rubric = st.session_state.rubric_data
            students = st.session_state.student_data_list
            
            total_ops = len(students) * len(rubric.get("questions", []))
            bar = st.progress(0)
            completed_ops = 0
            
            for student in students:
                student_evals = []
                stu_answers = {normalize_qno(a.get("question_no")): a for a in student.get("answers", [])}
                
                st.write(f"Grading **{student.get('student_metadata', {}).get('student_name', 'Unknown')}**...")
                
                for q_ref in rubric.get("questions", []):
                    qno = normalize_qno(q_ref.get("question_no"))
                    
                    # Match answer
                    # Handles duplicates poorly, just takes last one for now or first
                    # Improvements can be made here
                    ans_data = stu_answers.get(qno)
                    
                    ans_text = ""
                    ans_figs = ""
                    status = "Blank"
                    
                    if ans_data:
                        ans_text = ans_data.get("answer_text_plain", "")
                        ans_figs = ans_data.get("figure_summary_student", "")
                        status = ans_data.get("status", "Attempted")
                    
                    # Evaluate
                    eval_res = evaluate_single_answer(model, q_ref, ans_text, status, ans_figs)
                    eval_res = postprocess_evaluation(eval_res, q_ref.get("max_marks"))
                    
                    # Add metadata
                    # Add metadata with fallback for identification
                    s_meta = student.get("student_metadata", {})
                    s_name = s_meta.get("student_name")
                    if not s_name:
                        s_name = s_meta.get("roll_number")
                    if not s_name:
                        s_name = student.get("filename", "Unknown Student")
                        
                    eval_res["student_name"] = s_name
                    eval_res["student_roll"] = student.get("student_metadata", {}).get("roll_number")
                    eval_res["question_text"] = q_ref.get("question_text_plain")
                    eval_res["student_answer"] = ans_text
                    
                    all_evals.append(eval_res)
                    
                    completed_ops += 1
                    bar.progress(min(completed_ops / total_ops, 1.0))
            
            st.session_state.grading_results = all_evals
            st.success("Grading Complete!")
            
    # RESULTS DISPLAY
    if st.session_state.grading_results:
        # Separate results by student for clearer display
        df = pd.DataFrame(st.session_state.grading_results)
        
        # Prepare Rubric Data
        rubric_questions = st.session_state.rubric_data.get("questions", []) if st.session_state.rubric_data else []
        
        # Deduplicate questions for calculation
        unique_questions = {}
        for q in rubric_questions:
            qno = normalize_qno(q.get("question_no"))
            if qno not in unique_questions:
                unique_questions[qno] = q
        
        # 1. Try to get Total Marks from Metadata
        metadata_total = 0
        rubric_metadata = st.session_state.rubric_data.get("exam_metadata", {}) if st.session_state.rubric_data else {}
        try:
             metadata_total = float(rubric_metadata.get("total_marks", 0))
        except:
             metadata_total = 0
             
        # 2. Calculate Sum from Questions (Fallback & Check)
        calculated_total = 0
        for qno, q in unique_questions.items():
            try:
                calculated_total += float(q.get("max_marks", 0))
            except:
                pass
        
        # Decide which total to show
        display_total = metadata_total if metadata_total > 0 else calculated_total
        
        # Warning if mismatch (only if both exist and differ significantly)
        if metadata_total > 0 and calculated_total > 0 and abs(metadata_total - calculated_total) > 1.0:
             st.warning(f"⚠️ Note: Rubric Metadata states Total Marks = {metadata_total}, but sum of extracted questions = {calculated_total}. Using Metadata value for display.")
        
        # Get unique students
        student_names = df["student_name"].unique()
        
        for student_name in student_names:
            student_df = df[df["student_name"] == student_name]
            
            # Calculate Student Score
            student_score = 0
            for marks in student_df["marks_awarded"]:
                try:
                    student_score += float(marks)
                except:
                    pass
            
            st.divider()
            st.markdown(f"### 🎓 Results for: {student_name}")
            
            # Score Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Score", f"{student_score} / {display_total}")
            with col2:
                csv = student_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{student_name}_results.csv",
                    mime="text/csv",
                    key=f"csv_{student_name}"
                )
            with col3:
                 # Prepare nested JSON for download
                 student_json = student_df.to_json(orient="records", indent=2)
                 st.download_button(
                    label="📥 Download JSON",
                    data=student_json,
                    file_name=f"{student_name}_results.json",
                    mime="application/json",
                    key=f"json_{student_name}"
                 )

            # Detailed Breakdown (Readable UI)
            st.subheader("Detailed Breakdown")
            
            for index, row in student_df.iterrows():
                with st.expander(f"Q{row['question_no']}: {row['status']} ({row['marks_awarded']} / {row['max_marks']} Marks)"):
                    
                    # Columns for side-by-side view
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.markdown("**📝 Question:**")
                        st.markdown(row.get("question_text", "N/A"))
                        
                        st.markdown("**✍️ Student Answer:**")
                        st.markdown(row.get("student_answer", "N/A"))
                    
                    with c2:
                        st.markdown("**✅ Reference Criteria:**")
                        # Find matching rubric question for context if needed, 
                        # but we can rely on what's captured in eval_res if extended.
                        # For now, we show the result's feedback
                        
                        st.markdown("**🤖 AI Feedback:**")
                        if isinstance(row.get("stepwise_feedback"), list):
                            for step in row.get("stepwise_feedback"):
                                color = "green" if step.get('marks_awarded', 0) > 0 else "red"
                                st.markdown(f":{color}[• {step.get('feedback')} ({step.get('marks_awarded')}/{step.get('max_marks')})]")
                        else:
                            st.write(row.get("feedback"))
                            if row.get("error_details"):
                                st.error(f"Error Details: {row.get('error_details')}")
                    
                    # Raw JSON Viewer for this question (toggle)
                    st.caption("Raw Data")
                    st.json(row.to_dict())

            st.divider()

        # Global Downloads (All Students)
        st.subheader("📦 Bulk Actions")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            all_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download All Results (CSV)", all_csv, "all_results.csv", "text/csv")
        with col_b2:
            all_json = df.to_json(orient="records", indent=2)
            st.download_button("Download All Results (JSON)", all_json, "all_results.json", "application/json")
        
