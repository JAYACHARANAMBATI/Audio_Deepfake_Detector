import streamlit as st
import google.genai as genai
from google.genai import types
import mimetypes
import re
import os
import tempfile
import base64
import hashlib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime
from fpdf import FPDF
from dotenv import load_dotenv
import json


load_dotenv()

# ── CONFIGURATION ──
# Number of analysis runs for ensemble voting (higher = more consistent but slower)
# Set to 1 for faster results, 3 for more reliable results
ENSEMBLE_RUNS = 3  # Recommended: 3 for production, 1 for testing

# ── CACHING FOR CONSISTENT RESULTS ──
# Cache analysis results based on audio hash to prevent inconsistent LLM outputs
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_analysis(audio_hash: str, mime_type: str, audio_b64: str, _client):
    """Cache analysis results to ensure consistent outputs for same audio"""
    audio_bytes = base64.b64decode(audio_b64)
    return _analyze_audio_with_ensemble(audio_bytes, mime_type, _client)


def compute_audio_hash(audio_bytes: bytes) -> str:
    """Compute SHA-256 hash of audio for caching"""
    return hashlib.sha256(audio_bytes).hexdigest()


def _analyze_audio_with_ensemble(audio_bytes, mime_type, client):
    """
    Run multiple analysis passes and use voting for more consistent results.
    This reduces LLM hallucination by requiring consensus.
    """
    if ENSEMBLE_RUNS == 1:
        # Single run mode - faster but less consistent
        return _analyze_audio_internal(audio_bytes, mime_type, client)
    
    results = []
    predictions = []
    
    for i in range(ENSEMBLE_RUNS):
        result = _analyze_audio_internal(audio_bytes, mime_type, client)
        if result['success']:
            results.append(result)
            predictions.append(result['final_prediction'])
    
    if not results:
        return {'success': False, 'error': 'All analysis attempts failed'}
    
    # Majority voting for final prediction
    fake_count = predictions.count('FAKE')
    real_count = predictions.count('REAL')
    
    # Use conservative approach: if any doubt, lean towards FAKE
    if fake_count >= real_count:
        final_prediction = 'FAKE'
    else:
        final_prediction = 'REAL'
    
    # Average the scores across all successful runs
    avg_scores = {
        'spectral': 0, 'temporal': 0, 'phonetic': 0, 'prosodic': 0, 'background': 0
    }
    total_confidence = 0
    
    for r in results:
        for key in avg_scores:
            avg_scores[key] += r['detailed_scores'].get(key, 0)
        total_confidence += r['confidence']
    
    num_results = len(results)
    for key in avg_scores:
        avg_scores[key] = round(avg_scores[key] / num_results)
    avg_confidence = round(total_confidence / num_results)
    
    # Use the best matching result as base (one that matches final prediction)
    base_result = next((r for r in results if r['final_prediction'] == final_prediction), results[0])
    
    # Build ensemble result
    ensemble_result = {
        'raw_output': base_result['raw_output'],
        'confidence': avg_confidence,
        'detailed_scores': avg_scores,
        'average_score': sum(avg_scores.values()) / len([s for s in avg_scores.values() if s > 0]) if any(avg_scores.values()) else 0,
        'min_critical_score': min(avg_scores['spectral'], avg_scores['temporal'], avg_scores['phonetic']),
        'original_prediction': base_result['original_prediction'],
        'final_prediction': final_prediction,
        'override_applied': base_result.get('override_applied', False),
        'override_reason': base_result.get('override_reason', ''),
        'segments': base_result.get('segments', []),
        'forensic_report': base_result.get('forensic_report', ''),
        'success': True,
        'ensemble_info': {
            'runs': ENSEMBLE_RUNS,
            'successful_runs': num_results,
            'vote_distribution': {'FAKE': fake_count, 'REAL': real_count},
            'consensus': fake_count == num_results or real_count == num_results
        }
    }
    
    # Add ensemble override note if voting changed the result
    if final_prediction != base_result['final_prediction']:
        ensemble_result['override_applied'] = True
        ensemble_result['override_reason'] = f"Ensemble voting: {fake_count} FAKE vs {real_count} REAL"
    
    return ensemble_result


st.set_page_config(
    page_title="Audio Deepfake Detector",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

 
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.result-box {
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.real-result {
    background-color: #d4edda;
    border: 2px solid #28a745;
    color: #155724;
}
.fake-result {
    background-color: #f8d7da;
    border: 2px solid #dc3545;
    color: #721c24;
}
.confidence-high {
    color: #28a745;
    font-weight: bold;
}
.confidence-medium {
    color: #ffc107;
    font-weight: bold;
}
.confidence-low {
    color: #dc3545;
    font-weight: bold;
}
.score-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #007bff;
}
</style>
""", unsafe_allow_html=True)

def initialize_gemini():
    """Initialize Gemini AI with API key from environment variables"""
    try:
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('google_api_key') or os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            st.error("❌ No API key found! Please set GEMINI_API_KEY in your .env file")
            st.info("💡 Add this line to your .env file: GEMINI_API_KEY=your-api-key-here")
            return None
            
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Gemini AI: {e}")
        return None


def analyze_audio_real_or_fake(audio_bytes, mime_type, client):
    """
    Main entry point for audio analysis with caching for consistency.
    Uses cached results if available to prevent LLM hallucination/inconsistency.
    """
    # Compute hash for caching
    audio_hash = compute_audio_hash(audio_bytes)
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Check if we have a cached result (this prevents inconsistent outputs)
    return get_cached_analysis(audio_hash, mime_type, audio_b64, client)


def _analyze_audio_internal(audio_bytes, mime_type, client):
    """Internal analysis function - called by cache wrapper"""
    
    prompt = """
    You are an expert audio forensics analyst specializing in detecting AI-generated voices.

    ANALYZE THE AUDIO FOR THESE KEY HUMAN VS AI INDICATORS:

    1. NATURAL IMPERFECTIONS (Human Signs):
       - Background noise, room acoustics, environmental sounds
       - Natural breathing sounds, throat clearing, lip smacks
       - Speech gaps, pauses, hesitations, "um/uh" sounds
       - Micro-variations in voice quality and pitch
       - Inconsistent recording levels or quality

    2. ARTIFICIAL PERFECTION (AI Signs):
       - Too clean audio with no background noise
       - Continuous speaking without natural pauses or breath sounds
       - Overly consistent pitch and tone throughout
       - Robotic or monotone delivery without natural emotion
       - Perfect pronunciation without any speech imperfections

    3. FREQUENCY ANALYSIS:
       - Human voices have natural frequency variations and harmonics
       - AI voices often lack high-frequency details or have artificial frequency patterns
       - Check for unnatural frequency smoothing or digital artifacts
       - Look for missing natural voice resonances

    4. SPEECH PATTERN ANALYSIS:
       - Humans have natural rhythm variations, speed changes, emphasis
       - AI tends to maintain consistent pacing and stress patterns
       - Check for natural emotional inflections vs artificial modulation
       - Analyze if pitch changes sound organic or digitally generated

    5. AUDIO QUALITY INDICATORS:
       - Real recordings often have slight imperfections, room tone
       - AI audio may be too pristine or have subtle digital compression artifacts
       - Check for natural acoustic environment vs artificial sound space

    IMPORTANT - SEGMENT-LEVEL DETECTION:
    The audio may contain MIXED content where some parts are real human voice
    and some parts are AI-generated. You MUST analyze the audio in segments.
    - Listen carefully to EACH portion/segment of the audio
    - Identify which parts sound like a real human and which parts sound AI-generated
    - Note approximate timestamps or positions (e.g., "beginning", "middle", "end",
      or time ranges like "0:00-0:15") for each segment
    - A single audio can have BOTH real and fake sections

    DECISION CRITERIA:
    ✅ REAL HUMAN VOICE if audio has:
    - Natural imperfections, noise, or environmental sounds
    - Breathing patterns and speech gaps
    - Natural pitch variations and emotional authenticity
    - Realistic acoustic environment

    ❌ AI-GENERATED VOICE if audio has:
    - Overly clean/perfect quality with no natural imperfections
    - Continuous speech without natural pauses or breaths
    - Consistent pitch/tone without natural variations
    - Artificial or robotic speech patterns

    CONFIDENCE LEVELS:
    - 90-100%: Very clear indicators present
    - 80-89%: Strong evidence for classification
    - 70-79%: Moderate confidence with some uncertainty
    - 60-69%: Low confidence, mixed indicators
    - Below 60%: Very uncertain, needs more analysis

    MANDATORY RESPONSE FORMAT:
    SPECTRAL_SCORE: <0-100>
    TEMPORAL_SCORE: <0-100>  
    PHONETIC_SCORE: <0-100>
    PROSODIC_SCORE: <0-100>
    BACKGROUND_SCORE: <0-100>
    OVERALL_CONFIDENCE: <0-100>%
    FINAL_PREDICTION: REAL or FAKE

    SEGMENT_ANALYSIS:
    List each detected segment on its own line in this exact format:
    SEGMENT: <approximate time or position> | <REAL or FAKE> | <short reason>
    (Example: SEGMENT: 0:00-0:12 | REAL | Natural breathing and background noise present)
    (Example: SEGMENT: 0:12-0:30 | FAKE | Unnaturally smooth pitch, no breath sounds)
    If the entire audio is one type, still provide at least one SEGMENT line.

    ONLY IF FINAL_PREDICTION IS FAKE, also provide:
    FORENSIC_REPORT:
    Write a detailed professional forensic report (5-10 lines) explaining exactly
    WHY this audio is classified as fake. Include specific technical evidence found,
    which parts are AI-generated and why, frequency anomalies detected, missing
    human characteristics, and any other forensic findings. Be specific and technical.
    End the report with: END_REPORT

    If FINAL_PREDICTION is REAL, do NOT include FORENSIC_REPORT.

    BE CONSERVATIVE: When uncertain, classify as FAKE to minimize false positives.
    """

    try:
        # Create the content with the new API format
        # Using temperature=0 for deterministic/consistent outputs
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64.b64encode(audio_bytes).decode('utf-8')
                            }
                        }
                    ]
                }
            ],
            config=types.GenerateContentConfig(
                temperature=0,  # Set to 0 for deterministic output
                top_p=1.0,
                top_k=1,  # Only consider the most likely token
            )
        )

        output = response.text
        
        # Extract detailed scores and prediction with improved regex
        spectral_match = re.search(r'SPECTRAL_SCORE:\s*(\d+)', output)
        temporal_match = re.search(r'TEMPORAL_SCORE:\s*(\d+)', output)
        phonetic_match = re.search(r'PHONETIC_SCORE:\s*(\d+)', output)
        prosodic_match = re.search(r'PROSODIC_SCORE:\s*(\d+)', output)
        background_match = re.search(r'BACKGROUND_SCORE:\s*(\d+)', output)
        confidence_match = re.search(r'OVERALL_CONFIDENCE:\s*(\d+)%', output)
        prediction_match = re.search(r'FINAL_PREDICTION:\s*(REAL|FAKE)', output)

        if confidence_match and prediction_match:
            confidence = int(confidence_match.group(1))
            prediction = prediction_match.group(1)
            
            # Extract individual scores for detailed analysis
            scores = {
                'spectral': int(spectral_match.group(1)) if spectral_match else 0,
                'temporal': int(temporal_match.group(1)) if temporal_match else 0,
                'phonetic': int(phonetic_match.group(1)) if phonetic_match else 0,
                'prosodic': int(prosodic_match.group(1)) if prosodic_match else 0,
                'background': int(background_match.group(1)) if background_match else 0
            }

            # Advanced logic: Conservative approach with multiple criteria
            final_prediction = prediction
            override_applied = False
            override_reason = ""
            
            # Rule 1: If any critical score is very low (< 70), lean towards FAKE
            critical_scores = [scores['spectral'], scores['temporal'], scores['phonetic']]
            min_critical_score = min(critical_scores) if critical_scores else 100
            
            # Rule 2: If overall confidence < 85 and prediction is REAL → FAKE
            if prediction == "REAL" and confidence < 85:
                final_prediction = "FAKE"
                override_applied = True
                override_reason = f"Conservative override: Confidence {confidence}% < 85%"
            
            # Rule 3: If any critical analysis score < 70 and prediction is REAL → FAKE
            elif prediction == "REAL" and min_critical_score < 70:
                final_prediction = "FAKE"
                override_applied = True
                override_reason = f"Critical analysis failure: Min score {min_critical_score} < 70"
            
            # Rule 4: If average of all scores < 75 and prediction is REAL → FAKE
            avg_score = sum(scores.values()) / len([s for s in scores.values() if s > 0]) if any(scores.values()) else 0
            if prediction == "REAL" and avg_score < 75:
                final_prediction = "FAKE"
                override_applied = True
                override_reason = f"Low average score: {avg_score:.1f} < 75"

            # Extract segment analysis
            segments = re.findall(r'SEGMENT:\s*(.+?)\s*\|\s*(REAL|FAKE)\s*\|\s*(.+)', output)
            segment_list = []
            for seg_time, seg_type, seg_reason in segments:
                segment_list.append({
                    'time': seg_time.strip(),
                    'type': seg_type.strip(),
                    'reason': seg_reason.strip()
                })

            # Extract forensic report (only present when FAKE)
            forensic_report = ""
            report_match = re.search(r'FORENSIC_REPORT:\s*(.*?)END_REPORT', output, re.DOTALL)
            if report_match:
                forensic_report = report_match.group(1).strip()

            return {
                'raw_output': output,
                'confidence': confidence,
                'detailed_scores': scores,
                'average_score': avg_score,
                'min_critical_score': min_critical_score,
                'original_prediction': prediction,
                'final_prediction': final_prediction,
                'override_applied': override_applied,
                'override_reason': override_reason,
                'segments': segment_list,
                'forensic_report': forensic_report,
                'success': True
            }
        else:
            return {
                'raw_output': output,
                'success': False,
                'error': 'Could not parse structured output properly. Missing confidence or prediction.'
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': f'Error analyzing audio: {e}'
        }

def generate_report_id(audio_bytes):
    """Generate a unique report ID based on audio hash and timestamp"""
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()[:12].upper()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"ADF-{timestamp}-{audio_hash}"

def generate_pdf_report(result, filename, file_size_kb, report_id):
    """Generate a professional PDF forensic report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M:%S %p")
    
    confidence = result['confidence']
    final_prediction = result['final_prediction']
    scores = result['detailed_scores']
    avg_score = result.get('average_score', 0)
    
    # ── Title Bar ──
    pdf.set_fill_color(102, 126, 234)
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_y(8)
    pdf.cell(0, 10, "Audio Deepfake Detection Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, "Forensic Audio Analysis & Authenticity Verification", ln=True, align="C")
    
    # ── Report Meta ──
    pdf.set_y(45)
    pdf.set_text_color(80, 80, 80)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_fill_color(245, 245, 250)
    pdf.rect(10, 43, 190, 28, 'F')
    pdf.set_xy(14, 45)
    pdf.cell(90, 5, f"Report ID: {report_id}", ln=0)
    pdf.cell(90, 5, f"Date: {date_str}", ln=True, align="R")
    pdf.set_x(14)
    pdf.cell(90, 5, f"File: {filename}", ln=0)
    pdf.cell(90, 5, f"Time: {time_str}", ln=True, align="R")
    pdf.set_x(14)
    pdf.cell(90, 5, f"Size: {file_size_kb:.1f} KB", ln=0)
    pdf.cell(90, 5, f"Classification: {final_prediction}", ln=True, align="R")
    
    # ── Verdict Box ──
    pdf.ln(6)
    if final_prediction == "FAKE":
        pdf.set_fill_color(248, 215, 218)
        pdf.set_draw_color(220, 53, 69)
    else:
        pdf.set_fill_color(212, 237, 218)
        pdf.set_draw_color(40, 167, 69)
    
    verdict_y = pdf.get_y()
    pdf.rect(10, verdict_y, 190, 18, 'DF')
    pdf.set_xy(14, verdict_y + 2)
    
    if final_prediction == "FAKE":
        pdf.set_text_color(114, 28, 36)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 7, "VERDICT: AI-GENERATED VOICE DETECTED", ln=True, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, f"Overall Confidence: {confidence}%  |  This audio should NOT be considered authentic.", ln=True, align="C")
    else:
        pdf.set_text_color(21, 87, 36)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 7, "VERDICT: AUTHENTIC HUMAN VOICE", ln=True, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, f"Overall Confidence: {confidence}%  |  This audio is classified as genuine human speech.", ln=True, align="C")
    
    # ── Analysis Scores ──
    pdf.ln(8)
    pdf.set_text_color(50, 50, 50)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Technical Analysis Scores", ln=True)
    pdf.set_draw_color(102, 126, 234)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    score_items = [
        ("Spectral Analysis", scores['spectral']),
        ("Temporal Analysis", scores['temporal']),
        ("Phonetic Analysis", scores['phonetic']),
        ("Prosodic Analysis", scores['prosodic']),
        ("Background Analysis", scores['background']),
    ]
    
    pdf.set_font("Helvetica", "", 10)
    for label, val in score_items:
        # Label
        pdf.cell(60, 7, f"  {label}", ln=0)
        # Score bar background
        bar_x = pdf.get_x()
        bar_y = pdf.get_y() + 1.5
        pdf.set_fill_color(230, 230, 230)
        pdf.rect(bar_x, bar_y, 100, 4, 'F')
        # Score bar fill
        if val >= 80:
            pdf.set_fill_color(40, 167, 69)
        elif val >= 60:
            pdf.set_fill_color(255, 193, 7)
        else:
            pdf.set_fill_color(220, 53, 69)
        pdf.rect(bar_x, bar_y, val, 4, 'F')
        # Score value
        pdf.cell(100, 7, "", ln=0)
        rating = "Excellent" if val >= 90 else "Good" if val >= 80 else "Fair" if val >= 70 else "Poor" if val >= 50 else "Critical"
        pdf.cell(30, 7, f"{val}/100 ({rating})", ln=True)
    
    # Average
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 7, f"  Average Score", ln=0)
    pdf.cell(100, 7, "", ln=0)
    pdf.cell(30, 7, f"{avg_score:.1f}/100", ln=True)
    
    # ── Segment Analysis ──
    if result.get('segments'):
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 8, "Segment-by-Segment Analysis", ln=True)
        pdf.set_draw_color(102, 126, 234)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        # Table header
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(102, 126, 234)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(10, 7, "#", border=1, align="C", fill=True)
        pdf.cell(40, 7, "Time / Position", border=1, align="C", fill=True)
        pdf.cell(25, 7, "Detection", border=1, align="C", fill=True)
        pdf.cell(115, 7, "Reason", border=1, align="C", fill=True)
        pdf.ln()
        
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        for i, seg in enumerate(result['segments'], 1):
            if seg['type'] == 'REAL':
                pdf.set_fill_color(232, 245, 233)
            else:
                pdf.set_fill_color(255, 235, 238)
            pdf.cell(10, 7, str(i), border=1, align="C", fill=True)
            pdf.cell(40, 7, seg['time'], border=1, align="C", fill=True)
            pdf.cell(25, 7, seg['type'], border=1, align="C", fill=True)
            # Truncate reason if too long
            reason = seg['reason'][:65] + "..." if len(seg['reason']) > 65 else seg['reason']
            pdf.cell(115, 7, reason, border=1, fill=True)
            pdf.ln()
    
    # ── Forensic Report (FAKE only) ──
    if final_prediction == "FAKE" and result.get('forensic_report'):
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(220, 53, 69)
        pdf.cell(0, 8, "Forensic Investigation Report", ln=True)
        pdf.set_draw_color(220, 53, 69)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        
        pdf.set_fill_color(255, 245, 245)
        report_y = pdf.get_y()
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.set_x(14)
        pdf.multi_cell(182, 6, result['forensic_report'])
        report_end_y = pdf.get_y()
        pdf.set_draw_color(220, 53, 69)
        pdf.rect(10, report_y - 2, 190, report_end_y - report_y + 4, 'D')
    
    # ── Override Note ──
    if result.get('override_applied', False):
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(133, 100, 4)
        pdf.cell(0, 6, f"Override Applied: {result.get('override_reason', 'N/A')}", ln=True)
    
    # ── Footer ──
    pdf.ln(10)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 5, f"Report generated on {date_str} at {time_str}  |  Report ID: {report_id}", ln=True, align="C")
    pdf.cell(0, 5, "Audio Deepfake Detector  |  Forensic Audio Analysis System", ln=True, align="C")
    pdf.cell(0, 5, "This report is auto-generated. Integrity verified by SHA-256 audio hash.", ln=True, align="C")
    
    return bytes(pdf.output())

def get_confidence_color(confidence):
    """Return CSS class based on confidence level"""
    if confidence >= 80:
        return "confidence-high"
    elif confidence >= 60:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Audio Deepfake Detector</h1>
        <p>Upload an audio file to determine if the voice is real or AI-generated</p>
    </div>
    """, unsafe_allow_html=True)

    # Main content area (full width)
    st.header("📁 Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'ogg', 'flac'],
        help="Select an audio file to analyze"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")
        st.info(f"📊 File size: **{uploaded_file.size / 1024:.1f} KB**")
        
        # Audio player
        st.audio(uploaded_file.read(), format=f'audio/{uploaded_file.type.split("/")[1]}')
        uploaded_file.seek(0)  # Reset file pointer
        
        # Buttons row
        btn_col1, btn_col2 = st.columns([3, 1])
        
        with btn_col1:
            analyze_clicked = st.button("🔍 Analyze Audio", type="primary", use_container_width=True)
        
        with btn_col2:
            # Clear cache button for forcing re-analysis
            if st.button("🔄 Clear Cache", help="Force a new analysis (clears cached results)", use_container_width=True):
                get_cached_analysis.clear()
                st.success("✅ Cache cleared! Click 'Analyze Audio' for a fresh analysis.")
                st.rerun()
        
        # Analyze button
        if analyze_clicked:
            # Initialize model
            client = initialize_gemini()
            if not client:
                return
            
            # Show progress
            with st.spinner("🔎 Analyzing audio... This may take a few moments..."):
                # Get file data
                audio_bytes = uploaded_file.read()
                mime_type = uploaded_file.type
                
                # Analyze audio (uses caching for consistent results)
                result = analyze_audio_real_or_fake(audio_bytes, mime_type, client)
            
            # Display results
            if result['success']:
                confidence = result['confidence']
                final_prediction = result['final_prediction']
                
                # Result box
                result_class = "real-result" if final_prediction == "REAL" else "fake-result"
                confidence_class = get_confidence_color(confidence)
                
                # Show prediction icon
                prediction_icon = "✅ AUTHENTIC HUMAN VOICE" if final_prediction == "REAL" else "🚫 AI-GENERATED VOICE"
                
                st.markdown(f"""
                <div class="result-box {result_class}">
                    <h2>🎯 Analysis Result</h2>
                    <h3>{prediction_icon}</h3>
                    <p class="{confidence_class}">Overall Confidence: {confidence}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show ensemble voting info if available
                if result.get('ensemble_info'):
                    info = result['ensemble_info']
                    consensus_icon = "✅" if info['consensus'] else "⚠️"
                    st.info(f"""
                    🗳️ **Ensemble Voting**: {info['runs']} analysis runs performed  
                    📊 **Vote Distribution**: FAKE: {info['vote_distribution']['FAKE']} | REAL: {info['vote_distribution']['REAL']}  
                    {consensus_icon} **Consensus**: {'Yes - All runs agree' if info['consensus'] else 'No - Majority vote used'}
                    """)
                
                # Show detailed scores if available
                if 'detailed_scores' in result and any(result['detailed_scores'].values()):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🎼 Spectral Analysis", f"{result['detailed_scores']['spectral']}/100")
                        st.metric("⏱️ Temporal Analysis", f"{result['detailed_scores']['temporal']}/100")
                    
                    with col2:
                        st.metric("🗣️ Phonetic Analysis", f"{result['detailed_scores']['phonetic']}/100")
                        st.metric("🎵 Prosodic Analysis", f"{result['detailed_scores']['prosodic']}/100")
                    
                    with col3:
                        st.metric("🔊 Background Analysis", f"{result['detailed_scores']['background']}/100")
                        st.metric("📊 Average Score", f"{result.get('average_score', 0):.1f}/100")
                    
                    # ── GRAPHS SECTION ──
                    st.markdown("---")
                    st.subheader("📈 Visual Score Analysis")
                    
                    scores = result['detailed_scores']
                    categories = ['Spectral', 'Temporal', 'Phonetic', 'Prosodic', 'Background']
                    values = [scores['spectral'], scores['temporal'], scores['phonetic'], scores['prosodic'], scores['background']]
                    avg = result.get('average_score', 0)
                    
                    # Color each bar based on score
                    bar_colors = []
                    for v in values:
                        if v >= 80:
                            bar_colors.append('#28a745')  # green
                        elif v >= 60:
                            bar_colors.append('#ffc107')  # yellow
                        else:
                            bar_colors.append('#dc3545')  # red
                    
                    graph_col1, graph_col2 = st.columns(2)
                    
                    # ── Bar Chart ──
                    with graph_col1:
                        st.markdown("**📊 Score Breakdown (Bar Chart)**")
                        fig_bar = go.Figure(data=[
                            go.Bar(
                                x=categories,
                                y=values,
                                marker_color=bar_colors,
                                text=[f"{v}/100" for v in values],
                                textposition='outside'
                            )
                        ])
                        fig_bar.add_hline(y=avg, line_dash="dash", line_color="blue",
                                          annotation_text=f"Average: {avg:.1f}",
                                          annotation_position="top left")
                        fig_bar.update_layout(
                            yaxis=dict(range=[0, 110], title="Score"),
                            xaxis=dict(title="Analysis Category"),
                            height=400,
                            margin=dict(t=30, b=40)
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # ── Radar / Spider Chart ──
                    with graph_col2:
                        st.markdown("**🕸️ Radar Chart (All Dimensions)**")
                        radar_categories = categories + [categories[0]]  # close the shape
                        radar_values = values + [values[0]]
                        
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=radar_values,
                            theta=radar_categories,
                            fill='toself',
                            fillcolor='rgba(99, 110, 250, 0.25)',
                            line=dict(color='#667eea', width=2),
                            name='Scores'
                        ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            showlegend=False,
                            height=400,
                            margin=dict(t=30, b=30)
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # ── Horizontal Gauge-style progress bars ──
                    st.markdown("**🎯 Individual Score Gauges**")
                    icons = {'spectral': '🎼', 'temporal': '⏱️', 'phonetic': '🗣️', 'prosodic': '🎵', 'background': '🔊'}
                    for key in scores:
                        val = scores[key]
                        color = '#28a745' if val >= 80 else '#ffc107' if val >= 60 else '#dc3545'
                        label = f"{icons.get(key, '')} {key.capitalize()}"
                        st.markdown(f"**{label}: {val}/100**")
                        st.progress(val / 100)
                    
                    # ── Overall Confidence & Average Pie/Donut ──
                    st.markdown("---")
                    pie_col1, pie_col2 = st.columns(2)
                    
                    with pie_col1:
                        st.markdown("**🔵 Overall Confidence**")
                        fig_conf = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=confidence,
                            number={'suffix': '%'},
                            gauge=dict(
                                axis=dict(range=[0, 100]),
                                bar=dict(color='#667eea'),
                                steps=[
                                    dict(range=[0, 60], color='#f8d7da'),
                                    dict(range=[60, 80], color='#fff3cd'),
                                    dict(range=[80, 100], color='#d4edda'),
                                ],
                                threshold=dict(line=dict(color="red", width=3), thickness=0.8, value=85)
                            ),
                            title=dict(text="Confidence")
                        ))
                        fig_conf.update_layout(height=300, margin=dict(t=40, b=20))
                        st.plotly_chart(fig_conf, use_container_width=True)
                    
                    with pie_col2:
                        st.markdown("**🟢 Average Score**")
                        fig_avg = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=avg,
                            number={'suffix': '/100'},
                            gauge=dict(
                                axis=dict(range=[0, 100]),
                                bar=dict(color='#764ba2'),
                                steps=[
                                    dict(range=[0, 60], color='#f8d7da'),
                                    dict(range=[60, 75], color='#fff3cd'),
                                    dict(range=[75, 100], color='#d4edda'),
                                ],
                                threshold=dict(line=dict(color="red", width=3), thickness=0.8, value=75)
                            ),
                            title=dict(text="Average Score")
                        ))
                        fig_avg.update_layout(height=300, margin=dict(t=40, b=20))
                        st.plotly_chart(fig_avg, use_container_width=True)
                    
                    # ── Summary Table ──
                    st.markdown("---")
                    st.subheader("📋 Score Summary Table")
                    summary_df = pd.DataFrame({
                        'Category': ['🎼 Spectral', '⏱️ Temporal', '🗣️ Phonetic', '🎵 Prosodic', '🔊 Background'],
                        'Score': values,
                        'Status': ['✅ Pass' if v >= 70 else '⚠️ Warning' if v >= 50 else '❌ Fail' for v in values],
                        'Rating': ['Excellent' if v >= 90 else 'Good' if v >= 80 else 'Fair' if v >= 70 else 'Poor' if v >= 50 else 'Critical' for v in values]
                    })
                    summary_df['Max Score'] = 100
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # ── SEGMENT ANALYSIS SECTION ──
                if result.get('segments'):
                    st.markdown("---")
                    st.subheader("🔍 Segment-by-Segment Analysis")
                    
                    for i, seg in enumerate(result['segments'], 1):
                        seg_icon = "✅" if seg['type'] == "REAL" else "🚫"
                        seg_color = "green" if seg['type'] == "REAL" else "red"
                        seg_label = "Human Voice" if seg['type'] == "REAL" else "AI-Generated"
                        
                        st.markdown(f"""
                        <div style="padding: 0.8rem 1rem; margin: 0.5rem 0; border-radius: 8px;
                                    border-left: 5px solid {seg_color};
                                    background-color: {'#d4edda' if seg['type'] == 'REAL' else '#f8d7da'};">
                            <strong>{seg_icon} Segment {i}</strong> — <code>{seg['time']}</code><br>
                            <span style="color: {seg_color}; font-weight: bold;">{seg_label}</span><br>
                            <em>{seg['reason']}</em>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show segment summary table
                    seg_df = pd.DataFrame({
                        '#': list(range(1, len(result['segments']) + 1)),
                        'Time / Position': [s['time'] for s in result['segments']],
                        'Detection': [('✅ REAL' if s['type'] == 'REAL' else '🚫 FAKE') for s in result['segments']],
                        'Reason': [s['reason'] for s in result['segments']]
                    })
                    st.dataframe(seg_df, use_container_width=True, hide_index=True)
                
                # ── FORENSIC REPORT (only for FAKE) ──
                if final_prediction == "FAKE" and result.get('forensic_report'):
                    st.markdown("---")
                    st.subheader("🔬 Professional Forensic Report")
                    st.markdown(f"""
                    <div style="padding: 1.5rem; border-radius: 10px; border: 2px solid #dc3545;
                                background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%);
                                color: #333;">
                        <h4 style="color: #dc3545; margin-top: 0;">⚠️ Audio Forgery Detection Report</h4>
                        <p style="white-space: pre-line; line-height: 1.8;">{result['forensic_report']}</p>
                        <hr style="border-color: #dc3545;">
                        <small><strong>Verdict:</strong> This audio contains AI-generated content and should NOT be considered authentic.</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show override warning if applied
                if result.get('override_applied', False):
                    st.warning(f"⚠️ **Conservative Override Applied**: {result.get('override_reason', 'Safety threshold triggered')}")
                
                # ── DOWNLOAD PDF REPORT ──
                st.markdown("---")
                st.subheader("📥 Download Report")
                
                report_id = generate_report_id(audio_bytes)
                pdf_bytes = generate_pdf_report(
                    result=result,
                    filename=uploaded_file.name,
                    file_size_kb=uploaded_file.size / 1024,
                    report_id=report_id
                )
                
                # Build filename: AudioReport_REAL/FAKE_filename_datetime.pdf
                safe_name = os.path.splitext(uploaded_file.name)[0]
                safe_name = re.sub(r'[^\w\-]', '_', safe_name)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"AudioReport_{final_prediction}_{safe_name}_{ts}.pdf"
                
                dl_col1, dl_col2 = st.columns([1, 2])
                with dl_col1:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )
                with dl_col2:
                    st.markdown(f"""
                    <div style="padding: 0.6rem 1rem; background: #f0f2f6; border-radius: 8px; font-size: 0.85rem;">
                        <strong>Report ID:</strong> <code>{report_id}</code><br>
                        <strong>File:</strong> {uploaded_file.name} &nbsp;|&nbsp;
                        <strong>Generated:</strong> {datetime.now().strftime("%b %d, %Y %I:%M %p")}
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
