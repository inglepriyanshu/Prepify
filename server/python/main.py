import os
import pypdf
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import uvicorn
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sarvamai import SarvamAI
import json
from datetime import datetime
import base64
import edge_tts
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change "*" to your specific React URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SARVAM SETUP ---
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "YOUR_SARVAM_API_KEY")
sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

#Voice
VOICE_ID = "en-US-BrianNeural"

# --- GEMINI ROTATION SETUP ---
# Interview keys: use four separate API keys for normal interview chat.
# Feedback keys: use two separate API keys for feedback generation only.
# Old variables are still supported for compatibility.

GEMINI_CONFIGS = [
    {"api_key": os.environ.get("GEMINI_INTERVIEW_KEY_1", os.environ.get("GEMINI_KEY_1", "YOUR_KEY_1")), "model": "gemini-2.5-flash-lite"},
    {"api_key": os.environ.get("GEMINI_INTERVIEW_KEY_2", os.environ.get("GEMINI_KEY_2", "YOUR_KEY_2")), "model": "gemini-2.5-flash-lite"},
    {"api_key": os.environ.get("GEMINI_INTERVIEW_KEY_3", os.environ.get("GEMINI_KEY_3", "YOUR_KEY_3")), "model": "gemini-2.5-flash-lite"},
    {"api_key": os.environ.get("GEMINI_INTERVIEW_KEY_4", os.environ.get("GEMINI_KEY_4", "YOUR_KEY_4")), "model": "gemini-2.5-flash-lite"},
    {"api_key": os.environ.get("GEMINI_INTERVIEW_KEY_1", os.environ.get("GEMINI_KEY_1", "YOUR_KEY_1")), "model": "gemini-2.5-flash"},
    {"api_key": os.environ.get("GEMINI_INTERVIEW_KEY_2", os.environ.get("GEMINI_KEY_2", "YOUR_KEY_2")), "model": "gemini-2.5-flash"},
    {"api_key": os.environ.get("GEMINI_INTERVIEW_KEY_3", os.environ.get("GEMINI_KEY_3", "YOUR_KEY_3")), "model": "gemini-2.5-flash"},
    {"api_key": os.environ.get("GEMINI_INTERVIEW_KEY_4", os.environ.get("GEMINI_KEY_4", "YOUR_KEY_4")), "model": "gemini-2.5-flash"}
]

GEMINI_FEEDBACK_KEY_1 = os.environ.get("GEMINI_FEEDBACK_KEY_1")
GEMINI_FEEDBACK_KEY_2 = os.environ.get("GEMINI_FEEDBACK_KEY_2")
GEMINI_FEEDBACK_KEY = os.environ.get("GEMINI_FEEDBACK_KEY")

FEEDBACK_GEMINI_CONFIGS = []
for feedback_key in [GEMINI_FEEDBACK_KEY_1, GEMINI_FEEDBACK_KEY_2]:
    if feedback_key:
        FEEDBACK_GEMINI_CONFIGS.append({"api_key": feedback_key, "model": "gemini-2.5-flash-lite"})

for feedback_key in [GEMINI_FEEDBACK_KEY_1, GEMINI_FEEDBACK_KEY_2]:
    if feedback_key:
        FEEDBACK_GEMINI_CONFIGS.append({"api_key": feedback_key, "model": "gemini-2.5-flash"})

# Backward compatibility: single feedback key variable.
if GEMINI_FEEDBACK_KEY:
    FEEDBACK_GEMINI_CONFIGS.insert(0, {"api_key": GEMINI_FEEDBACK_KEY, "model": "gemini-2.5-flash-lite"})
    FEEDBACK_GEMINI_CONFIGS.append({"api_key": GEMINI_FEEDBACK_KEY, "model": "gemini-2.5-flash"})

if not FEEDBACK_GEMINI_CONFIGS:
    print("[WARNING] No feedback-specific Gemini keys configured; falling back to interview Gemini keys for feedback.")
    FEEDBACK_GEMINI_CONFIGS.extend(GEMINI_CONFIGS)

# State Trackers
active_chat_history = [] 
current_user_transcript = ""
system_instruction_text = ""
current_resume_text = ""
current_role = ""


async def generate_interviewer_audio(text):
    """Secretly streams Microsoft Azure Neural TTS directly into memory."""
    try:
        communicate = edge_tts.Communicate(text, VOICE_ID, rate="+25%")
        audio_data = b""
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
                
        # Encode the raw MP3 audio into a Base64 string so we can send it in the JSON response
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"TTS Engine Error: {e}")
        return None
    

def get_clean_transcript():
    """Extracts a clean, JSON-ready array from Gemini's complex history."""
    global active_chat_history
    clean_log = []
    
    for message in active_chat_history:
        role = message.role 
        text = message.parts[0].text if message.parts else ""
        
        # We don't need to save the giant hidden system prompt
        if role == "user" and "Hello, I am ready for the interview." in text:
            text = "[System Start Trigger]"
            
        clean_log.append({
            "speaker": "Interviewer" if role == "model" else "Candidate",
            "text": text
        })
        
    return clean_log


def format_transcript_for_feedback(transcript_entries):
    lines = []
    for item in transcript_entries:
        if item.get('text'):
            lines.append(f"{item.get('speaker')}: {item.get('text').strip()}")
    return "\n".join(lines).strip()


def extract_qa_pairs(transcript_entries):
    qa_pairs = []
    current_question = None
    current_answer_lines = []

    def append_pair():
        answer_text = ' '.join(current_answer_lines).strip()
        if answer_text:
            qa_pairs.append({
                'question': current_question,
                'answer': answer_text
            })

    for item in transcript_entries:
        speaker = item.get('speaker')
        text = item.get('text', '').strip()
        if not text:
            continue

        if speaker == 'Interviewer':
            if current_question is not None:
                append_pair()
            current_question = text
            current_answer_lines = []
        elif speaker == 'Candidate':
            if current_question is not None:
                current_answer_lines.append(text)

    if current_question is not None:
        append_pair()

    return qa_pairs


def build_feedback_prompt(resume_text: str, role_name: str, transcript_text: str, qa_pairs) -> str:
    qa_section = ''
    if qa_pairs:
        qa_lines = []
        for index, pair in enumerate(qa_pairs, start=1):
            qa_lines.append(f"{index}. Question: {pair['question']}\nAnswer: {pair['answer']}")
        qa_section = '\n\nAnswered interview question-answer pairs:\n' + '\n\n'.join(qa_lines)

    return f"""
You are an expert technical interview coach. Use the candidate resume, the interview transcript, and the extracted answered question-answer pairs to generate structured feedback for the interview.

Candidate role: {role_name}

Candidate resume:
{resume_text}

Interview transcript:
{transcript_text}{qa_section}

The question-answer pairs above include only answered questions. Do not include any unanswered or partially answered questions in "question_feedback".

When presenting the candidate's answer as "your_answer", correct only obvious speech-to-text transcription mistakes and preserve the original meaning. Do not invent new information.

If there is at least one answered question in the transcript, you must provide one question_feedback object for each answered question. Only return an empty "question_feedback" array if there are zero answered questions.

Please return ONLY valid JSON with the following shape:
{{
  "overall_rating": 1-10,
  "overall_comments": "...",
  "question_feedback": [
    {{
      "question": "...",
      "your_answer": "...",
      "expected_answer": "...",
      "improvements": "...",
      "rating": 1-10
    }}
  ]
}}

If the interview transcript contains one or more answered question-answer pairs, include each pair in "question_feedback". If the transcript cannot be clearly separated into question-answer pairs, return an empty "question_feedback" array and explain why in "overall_comments".

Do not add any markdown, backticks, or extra text outside the JSON object.
""".strip()


def send_to_gemini_text_with_failover(prompt: str):
    """Tries keys and models in order until one succeeds for text-only requests."""
    for config in FEEDBACK_GEMINI_CONFIGS:
        print(f"Feedback request using model {config['model']} with key ending in ...{config['api_key'][-4:]}" )
        try:
            client = genai.Client(api_key=config['api_key'])
            chat_config = types.GenerateContentConfig()
            chat_session = client.chats.create(model=config['model'], config=chat_config)
            response = chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            print(f"Feedback generation failed with {config['model']}: {e}")
            if "503" in str(e) or "429" in str(e):
                time.sleep(1)
                continue
            raise
    raise HTTPException(status_code=503, detail="All feedback AI failovers exhausted.")


def parse_feedback_json(raw_text: str):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        first = raw_text.find('{')
        last = raw_text.rfind('}')
        if first != -1 and last != -1 and last > first:
            sub = raw_text[first:last+1]
            try:
                return json.loads(sub)
            except json.JSONDecodeError:
                pass
    return {"raw_text": raw_text.strip()}

def extract_text_from_pdf(pdf_path):
    extracted_text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"

        extracted_text = extracted_text.strip()
        if not extracted_text:
            print(f"[PDF EXTRACT] No text extracted from PDF: {pdf_path}")
            return "Resume data unavailable."

        print(f"[PDF EXTRACT] Extracted {len(extracted_text)} characters from PDF: {pdf_path}")
        return extracted_text
    except Exception as err:
        print(f"[PDF EXTRACT ERROR] {err} | path={pdf_path}")
        return "Resume data unavailable."

# ... (Keep your MASTER_PROMPT and ML_ENGINEER_GUIDELINE variables here) ...
MASTER_PROMPT = """You are an expert, professional technical interviewer.
The candidate's selected role is: {role_name}

Here is the candidate's resume:
{resume_text}

Here are the guidelines for the role they are applying for:
{role_guideline}

YOUR INSTRUCTIONS:
1. Conduct a realistic, back-and-forth technical interview.
2. ONLY ask questions relevant to the selected role. Do not ask any questions for roles other than the selected role.
3. Ask ONLY ONE question at a time. NEVER output a list of questions. Wait for the candidate's response before continuing.
4. THE OPENING: Start the interview by welcoming the candidate and asking them to briefly introduce themselves and their background. 
5. THE TRANSITION: Once they introduce themselves, acknowledge their introduction and ask your first targeted technical question based on a specific project or skill from their resume.
6. CROSS-QUESTIONING: Listen carefully to the candidate's answers. Ask follow-up questions to probe their actual depth of knowledge, trade-offs, and reasoning.
7. TIME MANAGEMENT & PIVOTING (CRITICAL CONSTRAINT): Limit your cross-questioning on any individual project, past role, or specific skill to a maximum of 3 to 4 questions. Once you hit this limit, you MUST explicitly transition to a completely different section of their resume. 
8. Keep your tone professional, encouraging, but rigorous. Do not provide answers or break character.
9. CONCISENESS (CRITICAL SYSTEM RULE): Keep your responses short, conversational, and directly to the point. NEVER yap or give long monologues. Limit your replies and questions to a maximum of 2 to 3 sentences.
"""

ML_ENGINEER_GUIDELINE = """
Target Role: Machine Learning Engineer (Mid-Level)

Core Competencies to Evaluate:
1. ML Fundamentals: Algorithm selection, bias-variance tradeoff, and evaluation metrics (Precision, Recall, F1, ROC-AUC) in imbalanced datasets.
2. Deep Learning & Advanced Models: Neural network architectures (Transformers, CNNs), attention mechanisms, fine-tuning strategies (LoRA, PEFT), and optimization.
3. MLOps & Production: Model serving (latency vs. throughput), containerization, monitoring (data drift, concept drift), and handling production bottlenecks.
4. Data/Feature Engineering: Data pipelines, handling missing data, and scaling data processing.

Interview Strategy & Focus:
- DO NOT ask for basic textbook definitions (e.g., "What is a neural network?").
- DO ask about architectural decisions, trade-offs, and edge cases (e.g., "Why did you choose a Transformer over an LSTM for this specific latency constraint?").
- Probe their understanding of how their code operates in a real-world, scalable production environment, not just in a local Jupyter Notebook.
- If they mention a specific framework or tool in their resume, ask them about its limitations.
"""

SOFTWARE_ENGINEER_GUIDELINE = """
Target Role: Software Engineer (Mid-Level)

Core Competencies to Evaluate:
1. System Design: Building scalable services, API design, microservices vs monolith trade-offs, and fault tolerance.
2. Algorithms & Data Structures: Time/space complexity, problem solving, and efficient implementation choices.
3. Testing & Quality: Unit testing, integration testing, CI/CD, and debugging practices.
4. Code Architecture: Clean code, modular design, dependency management, and working with cross-functional teams.

Interview Strategy & Focus:
- Focus on real-world engineering decisions and trade-offs, not abstract definitions.
- Ask about how the candidate designs maintainable, testable systems under constraints.
- Probe their ability to reason about edge cases, failure modes, and system performance.
"""

FRONTEND_ENGINEER_GUIDELINE = """
Target Role: Front-end Engineer (Mid-Level)

Core Competencies to Evaluate:
1. Web Fundamentals: HTML semantics, CSS layout, browser rendering, and responsive UI implementation.
2. JavaScript & Frameworks: React/Angular/Vue component design, state management, hooks, and client-side routing.
3. Performance & Accessibility: Optimizing rendering, reducing bundle size, and building accessible interfaces.
4. Debugging & Testing: Troubleshooting browser issues, cross-browser compatibility, and component testing.

Interview Strategy & Focus:
- Ask about component architecture, state management, and front-end performance trade-offs.
- Probe their understanding of the browser event loop, reflows, and optimization strategies.
- Evaluate whether they can translate resume projects into practical UI solutions.
"""

BACKEND_ENGINEER_GUIDELINE = """
Target Role: Back-end Engineer (Mid-Level)

Core Competencies to Evaluate:
1. API Design: REST/GraphQL, authentication, versioning, and backward compatibility.
2. Data Storage & Querying: Database selection, indexing, transactions, and schema design.
3. Scalability & Reliability: Caching, load balancing, monitoring, and failure recovery.
4. Security & Infrastructure: Secure data handling, authorization, service orchestration, and deployment.

Interview Strategy & Focus:
- Focus on API reliability, data consistency, and system integration.
- Ask about trade-offs between different persistence and scaling strategies.
- Probe their ability to reason about production deployments, observability, and resiliency.
"""

ROLE_GUIDELINES = {
    'Machine Learning Engineer': ML_ENGINEER_GUIDELINE,
    'Software Engineer': SOFTWARE_ENGINEER_GUIDELINE,
    'Front-end Engineer': FRONTEND_ENGINEER_GUIDELINE,
    'Back-end Engineer': BACKEND_ENGINEER_GUIDELINE,
}


def send_to_gemini_with_failover(user_message: str):
    """Tries keys and models in order until one succeeds."""
    global active_chat_history, system_instruction_text
    
    last_error = None
    
    for config in GEMINI_CONFIGS:
        print(f"Attempting to use model {config['model']} with key ending in ...{config['api_key'][-4:]}")
        try:
            # 1. Spin up a fresh client with the current config's key
            client = genai.Client(api_key=config['api_key'])
            
            # 2. Rebuild the chat session with our saved history
            chat_config = types.GenerateContentConfig(system_instruction=system_instruction_text)
            chat_session = client.chats.create(
                model=config['model'], 
                config=chat_config,
                history=active_chat_history
            )
            
            # 3. Send the message
            response = chat_session.send_message(user_message)
            
            # 4. If successful, save the new history so we don't lose it
            active_chat_history = chat_session.get_history()
            return response.text
            
        except Exception as e:
            last_error = str(e)
            print(f"Failed. Error: {last_error}")
            if "503" in last_error or "429" in last_error:
                print("Rotating to next backup config...")
                time.sleep(1) # Brief pause before hammering the next API
                continue
            else:
                # If it's a structural error (not a rate limit/server down), stop trying
                break 

    # If the loop finishes and all configs failed
    raise HTTPException(status_code=503, detail="All AI failovers exhausted. Please try again.")




@app.post("/api/start-interview")
async def start_interview(resume_file: UploadFile = File(...), role: str = Form("Machine Learning Engineer")):
    global active_chat_history, system_instruction_text, current_user_transcript

    selected_role = role.strip() or "Machine Learning Engineer"
    temp_resume_path = None

    if resume_file is None:
        raise HTTPException(status_code=400, detail="Resume file is required.")

    try:
        resume_filename = resume_file.filename
        temp_resume_path = f"temp_resume_{int(time.time())}_{resume_file.filename}"
        with open(temp_resume_path, "wb+") as file_object:
            file_object.write(await resume_file.read())
        resume_text = extract_text_from_pdf(temp_resume_path)
        print(f"[START_INTERVIEW] Received uploaded resume '{resume_file.filename}' for role '{selected_role}'.")

        role_guideline = ROLE_GUIDELINES.get(selected_role, ROLE_GUIDELINES["Machine Learning Engineer"])
        system_instruction_text = MASTER_PROMPT.format(
            role_name=selected_role,
            resume_text=resume_text,
            role_guideline=role_guideline,
        )
        current_resume_text = resume_text
        current_role = selected_role

        # Reset state for a new interview
        active_chat_history = []
        current_user_transcript = ""

        reply_text = send_to_gemini_with_failover("Hello, I am ready for the interview. Please begin with the first question.")
        audio_base64 = await generate_interviewer_audio(reply_text)

        return {
            "reply": reply_text,
            "audio_data": audio_base64,
            "selected_role": selected_role,
            "resume_filename": resume_filename,
        }
    finally:
        if temp_resume_path and os.path.exists(temp_resume_path):
            os.remove(temp_resume_path)

@app.post("/api/audio-chunk")
async def process_audio_chunk(audio_file: UploadFile = File(...), is_final: str = Form(...)):
    """Receives 25s chunks, transcribes them, and triggers Gemini only on the final chunk."""
    global current_user_transcript
    
    temp_file_path = f"temp_{audio_file.filename}"
    try:
        with open(temp_file_path, "wb+") as file_object:
            file_object.write(await audio_file.read())

        # Transcribe this specific 25s chunk
        with open(temp_file_path, "rb") as f:
            transcript_response = sarvam_client.speech_to_text.transcribe(
                file=f,
                model="saaras:v3",
                mode="transcribe" 
            )
            
        if transcript_response.transcript:
            current_user_transcript += transcript_response.transcript + " "
            print(f"[Chunk Received] Current text: {current_user_transcript}")

        # If the user is still speaking, just return a success status
        if is_final == "false":
            return {"status": "chunk_processed", "current_text": current_user_transcript}

        # --- IF THIS IS THE FINAL CHUNK, SEND TO GEMINI ---
        if is_final == "true":
            final_text = current_user_transcript.strip()
            
            if not final_text:
                return {"user_text": "", "reply": "I didn't quite catch that. Could you repeat?"}

            # 1. Get the text reply from the failover engine
            reply_text = send_to_gemini_with_failover(final_text)
            
            # 2. Generate the voice!
            audio_base64 = await generate_interviewer_audio(reply_text)
            
            current_user_transcript = "" 
            
            # 3. Send the complete package back to the frontend
            return {
                "user_text": final_text, 
                "reply": reply_text,
                "audio_data": audio_base64
            }

    except Exception as e:
        print(f"\n--- AUDIO PROCESSING ERROR ---\n{str(e)}\n")
        raise HTTPException(status_code=500, detail="Failed to process audio chunk.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/end-interview")
def end_interview():
    global active_chat_history, current_user_transcript, current_resume_text, current_role
    
    if not active_chat_history:
        return {"message": "No interview data to save."}
        
    final_transcript = get_clean_transcript()
    transcript_text = format_transcript_for_feedback(final_transcript)
    qa_pairs = extract_qa_pairs(final_transcript)
    
    print(f"[DEBUG] Transcript entries: {len(final_transcript)}")
    print(f"[DEBUG] QA pairs extracted: {len(qa_pairs)}")
    if qa_pairs:
        print(f"[DEBUG] First QA pair: {qa_pairs[0]}")
    print(f"[DEBUG] Sample transcript: {transcript_text[:500]}...")
    
    feedback_prompt = build_feedback_prompt(
        resume_text=current_resume_text or "Resume not available.",
        role_name=current_role or "Unknown Role",
        transcript_text=transcript_text,
        qa_pairs=qa_pairs,
    )
    
    print(f"[DEBUG] Feedback prompt length: {len(feedback_prompt)} chars")
    
    feedback_response = send_to_gemini_text_with_failover(feedback_prompt)
    
    print(f"[DEBUG] Raw Gemini response: {feedback_response[:1000]}...")
    
    feedback_data = parse_feedback_json(feedback_response)
    
    print(f"[DEBUG] Parsed feedback: {feedback_data}")
    
    saved_role = current_role

    # Wipe the server memory clean for the next candidate
    active_chat_history = []
    current_user_transcript = ""
    current_resume_text = ""
    current_role = ""
    
    return {
        "status": "success",
        "feedback": feedback_data,
        "role": saved_role,
        "raw_output": feedback_response if isinstance(feedback_data, dict) and "raw_text" in feedback_data else None
    }

if __name__ == "__main__":
    print("Server starting at http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)