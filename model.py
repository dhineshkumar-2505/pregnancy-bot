from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import datetime  
import email.utils  
import asyncio  
from gtts import gTTS  
import os  
import random  

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Provide a helpful response based on the information provided. 
If the answer isn't clear, simply state that you don't know rather than attempting to guess.

Background Information: {context}
User's Query: {question}

Respond only with a useful and accurate answer below.
Helpful answer:
"""

# Email credentials
DOCTOR_EMAIL = "haritha.appaswamy@gmail.com"
SENDER_EMAIL = "dharpadma2004@gmail.com"
SENDER_PASSWORD = "dzyd qery dhud ovro"
IMAP_SERVER = "imap.gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Global variable for query timestamp
query_timestamp = None

# Function to convert text to speech
def text_to_speech(text, filename="response.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    os.system(f"start {filename}")  # For Windows, use "afplay" for Mac, or "mpg321" for Linux

# Function to classify the question
def classify_question(question):
    critical_keywords = [
        "bleeding", "spotting", "severe pain", "contractions", "emergency",
        "urgent", "water broke", "swelling", "high blood pressure", "shortness of breath",
        "chest pain", "faint", "nausea", "vomiting", "fetal movement", "infection"
    ]
    if any(keyword in question.lower() for keyword in critical_keywords):
        return "critical"
    return "basic"

# Function to detect emotional distress
def detect_emotional_distress(message):
    distress_keywords = ["depressed", "suicidal", "hopeless", "crying a lot", "alone", "can't cope"]
    general_stress_keywords = ["stressed", "overwhelmed", "anxious", "scared", "worried", "tired"]

    if any(keyword in message.lower() for keyword in distress_keywords):
        return "urgent"
    if any(keyword in message.lower() for keyword in general_stress_keywords):
        return "normal"

    return None

# Function to notify the doctor via email
def notify_doctor(question):
    subject = "Critical Query Received"
    body = f"A critical query has been received:\n\n{question}"

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = DOCTOR_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, DOCTOR_EMAIL, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")

# Function to fetch the doctor's reply
def fetch_doctor_reply(query_time):
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(SENDER_EMAIL, SENDER_PASSWORD)
        mail.select("inbox")

        status, messages = mail.search(None, f'FROM "{DOCTOR_EMAIL}"')
        if status != "OK" or not messages[0]:
            return None

        for email_id in reversed(messages[0].split()):
            status, data = mail.fetch(email_id, "(RFC822)")
            if status != "OK" or not data:
                continue

            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)

            email_date = msg["Date"]
            email_timestamp = email.utils.parsedate_to_datetime(email_date)

            if email_timestamp > query_time:
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(errors="ignore").strip()
                            return body
                else:
                    return msg.get_payload(decode=True).decode(errors="ignore").strip()

        return None
    except Exception as e:
        print(f"Failed to fetch doctor's reply. Error: {e}")
        return None

# Wait for the doctor's reply
async def wait_for_doctor_reply(query_time):
    await cl.Message(content="Waiting for the doctor's reply...").send()
    for attempt in range(10):
        reply = fetch_doctor_reply(query_time)
        if reply:
            await cl.Message(content=f"Doctor's Reply:\n{reply.strip()}").send()
            text_to_speech(reply.strip(), "doctor_reply.mp3")
            return  
        await cl.Message(content=f"Polling attempt {attempt + 1}: No reply yet. Retrying in 30 seconds...").send()
        await asyncio.sleep(30)

    await cl.Message(content="No reply from the doctor yet. Please check back later.").send()

# QA Bot setup
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def load_llm():
    return CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0 (1).bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, qa_prompt, db)

@cl.on_chat_start
async def start():
    chain = qa_bot()
    await cl.Message(content="Hi, Welcome to Pregnancypal, A Pregnancy Bot. What is your query?").send()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    global query_timestamp
    query_timestamp = datetime.datetime.now(datetime.timezone.utc)

    distress_level = detect_emotional_distress(message.content)
    if distress_level:
        if distress_level == "urgent":
            response = "I'm really sorry you're feeling this way. 💔 You're not alone. Please reach out to a mental health professional or call a support line. If you're in immediate danger, please seek help. 💖"
        else:
            affirmations = [
                "Pregnancy can be overwhelming, but remember, you are strong and capable. 💖",
                "Take a deep breath. You're growing a beautiful life, and you're doing amazing! 🌸",
            ]
            response = random.choice(affirmations)
        
        await cl.Message(content=response).send()
        text_to_speech(response, "distress_response.mp3")
        return

    query_type = classify_question(message.content)
    if query_type == "critical":
        await cl.Message(content="A doctor will get back to you shortly.").send()
        notify_doctor(message.content)
        await wait_for_doctor_reply(query_timestamp)
    else:
        chain = cl.user_session.get("chain")
        res = await chain.ainvoke(message.content)
        await cl.Message(content=res["result"]).send()