from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import TextSendMessage, QuickReply, QuickReplyButton, MessageAction
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
import json

# OLLAMA API settings
OLLAMA_API_URL = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "test")

def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

def save_user_info(uid, name):
    query = '''
    MERGE (u:User {uid: $uid})
    SET u.name = $name
    '''
    run_query(query, parameters={'uid': uid, 'name': name})

def get_user_name(uid):
    query = '''
    MATCH (u:User {uid: $uid})
    RETURN u.name AS name
    '''
    result = run_query(query, parameters={'uid': uid})
    return result[0]['name'] if result else None

def compute_response(sentence):
    if not sentence.strip():
        return None

    greeting_corpus = list(set(record['question'] for record in run_query('MATCH (n:Question) RETURN n.question as question;')))
    
    if not greeting_corpus:
        print("Greeting corpus is empty")
        return None

    greeting_vec = model.encode(greeting_corpus, convert_to_tensor=True, normalize_embeddings=True)
    ask_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)

    greeting_scores = util.cos_sim(greeting_vec, ask_vec)
    max_greeting_score_index = np.argmax(greeting_scores.cpu().numpy())
    if greeting_scores[max_greeting_score_index] > 0.8:
        match_greeting = greeting_corpus[max_greeting_score_index]
        my_cypher = f"MATCH (n:Question) WHERE n.question = '{match_greeting}' RETURN n.msg_reply AS reply"
        results = run_query(my_cypher)
        return results[0]['reply'] if results else None

    return None

def log_question_answer(question, answer):
    cypher_query = '''
    MATCH (q:Question {text: $question}) 
    MATCH (a:Answer {text: $answer}) 
    MERGE (q)-[:HAS_ANSWER]->(a)
    '''
    run_query(cypher_query, parameters={"question": question, "answer": answer})

def check_previous_question(question):
    cypher_query = '''
    MATCH (q:Question {text: $question}) 
    MATCH (a:Answer) 
    RETURN a.text AS answer
    '''
    result = run_query(cypher_query, parameters={"question": question})
    return result[0]['answer'] if result else None

def is_similar_query(user_query, expected_queries):
    user_vec = model.encode(user_query, convert_to_tensor=True, normalize_embeddings=True)
    for expected in expected_queries:
        expected_vec = model.encode(expected, convert_to_tensor=True, normalize_embeddings=True)
        score = util.cos_sim(user_vec, expected_vec)
        if score > 0.7:
            return True
    return False

def remove_endings(text):
    endings = ["ครับ", "ค่ะ", "น้ะ", "นะ", "นะจ้ะ"]
    for ending in endings:
        text = text.replace(ending, "")
    return text.strip()

def save_response(uid, answer_text, response_msg):
    query = '''
    MATCH (u:User {uid: $uid})
    CREATE (a:Answer {text: $answer_text})
    CREATE (r:Response {text: $response_msg})
    CREATE (u)-[:useranswer]->(a)
    CREATE (a)-[:response]->(r)
    '''
    parameters = {
        'uid': uid,
        'answer_text': answer_text,
        'response_msg': response_msg
    }
    run_query(query, parameters)
    

app = Flask(__name__)

with open('usr_champ.txt', 'r') as file:
    lines = file.readlines()
    channel_access_token = lines[0].strip()
    channel_secret = lines[1].strip()

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# ---- From Second Code: Quick Reply Functions -----

def quick_reply_menu(line_bot_api, tk, user_id, msg):
    quick_reply_button = QuickReplyButton(
        action=MessageAction(label="เมนู", text="เมนู")
    )
    quick_reply = QuickReply(
        items=[quick_reply_button]
    )
    line_bot_api.reply_message(tk, [TextSendMessage(text="เลือกเมนูที่ต้องการ", quick_reply=quick_reply)])

# ---- End of Quick Reply ----

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)

        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        uid = json_data['events'][0]['source']['userId']

        msg = remove_endings(msg)

        if "ชื่อ" in msg and "อะไร" in msg:
            user_name = get_user_name(uid)
            if user_name:
                line_bot_api.reply_message(tk, TextSendMessage(text=f"ชื่อของคุณคือ {user_name} ครับ"))
            else:
                line_bot_api.reply_message(tk, TextSendMessage(text="ขอโทษครับ ฉันไม่ทราบชื่อของคุณ"))

        elif "ชื่อ" in msg:
            name = msg.split("ชื่อ")[-1].strip()
            if name:
                save_user_info(uid, name)
                line_bot_api.reply_message(tk, TextSendMessage(text=f"ขอบคุณที่แนะนำตัวครับ {name}"))
            else:
                line_bot_api.reply_message(tk, TextSendMessage(text="ไม่สามารถระบุชื่อได้ กรุณาระบุชื่อของคุณครับ"))

        # Check for quick reply menu request
        if msg in ["เมนู", "menu", "Menu"]:
            quick_reply_menu(line_bot_api, tk, uid, msg)

        response_msg = compute_response(msg)

        if response_msg:
            line_bot_api.reply_message(tk, TextSendMessage(text=response_msg + " ครับ"))
            save_response(uid, msg, response_msg)  # บันทึกคำตอบ
        else:
            previous_answer = check_previous_question(msg)
            if previous_answer:
                line_bot_api.reply_message(tk, TextSendMessage(text=previous_answer + " ครับ"))
            else:
                payload = {
                    "model": "supachai/llama-3-typhoon-v1.5",
                    "prompt": f"ผู้ถามชื่อ คุณ{get_user_name(uid)} ตอบสั้นๆไม่เกิน 20 คำ เกี่ยวกับ '{msg}'",
                    "stream": False
                }
                response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
                if response.status_code == 200:
                    decoded_text = response.json().get("response", "")
                    line_bot_api.reply_message(tk, TextSendMessage(text=decoded_text + " ครับ\n.....คำตอบจาก Ollama..."))
                    log_question_answer(msg, decoded_text)
                    save_response(uid, msg, decoded_text)  # บันทึกคำตอบ
                else:
                    print(f"Failed to get a response from Ollama: {response.status_code}, {response.text}")
                    line_bot_api.reply_message(tk, TextSendMessage(text="เกิดข้อผิดพลาดในการติดต่อ LLaMA"))

    except InvalidSignatureError:
        print("Invalid signature.")
    except Exception as e:
        print("Error:", e)
        print(body)
    return 'OK'


if __name__ == "__main__":
    app.run(port=5000)
