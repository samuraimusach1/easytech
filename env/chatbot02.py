from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi
from linebot.v3.webhook import WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, QuickReply, QuickReplyButton, MessageAction
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
import json
from bs4 import BeautifulSoup
import re
from selenium import webdriver
import chromedriver_autoinstaller

# Setup Chrome options and install chromedriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chromedriver_autoinstaller.install()

# Constants
OLLAMA_API_URL = "http://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "test")

# Database query functions
def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            return [record for record in session.run(query, parameters)]

def save_user_info(uid, name):
    run_query('MERGE (u:User {uid: $uid}) SET u.name = $name', {'uid': uid, 'name': name})


def get_user_name(uid):
    query = '''
    MATCH (u:User {uid: $uid})
    RETURN u.name AS name
    '''
    result = run_query(query, parameters={'uid': uid})
    return result[0]['name'] if result else None

def log_chat_history(uid, message, reply):
    run_query('MATCH (u:User {uid: $uid}) CREATE (c:Chat {message: $message, reply: $reply, timestamp: timestamp()}) CREATE (u)-[:SENT]->(c)', {'uid': uid, 'message': message, 'reply': reply})

def save_response(uid, answer_text, response_msg):
    run_query('MATCH (u:User {uid: $uid}) CREATE (a:Answer {text: $answer_text}) CREATE (r:Response {text: $response_msg}) CREATE (u)-[:useranswer]->(a) CREATE (a)-[:response]->(r)', {'uid': uid, 'answer_text': answer_text, 'response_msg': response_msg})
def clean_price(price_str):
    cleaned_price = re.sub(r'[^\d]', '', price_str)
    return int(cleaned_price)


def compute_response(sentence):
    greeting_corpus = list(set(record['name'] for record in run_query('MATCH (n:Greeting) RETURN n.name as name;')))
    greeting_vec = model.encode(greeting_corpus, convert_to_tensor=True, normalize_embeddings=True)
    ask_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    greeting_scores = util.cos_sim(greeting_vec, ask_vec)
    
    if (max_index := np.argmax(greeting_scores.cpu().numpy())) and greeting_scores[max_index] > 0.6:
        match_greeting = greeting_corpus[max_index]
        results = run_query(f"MATCH (n:Greeting) WHERE n.name = '{match_greeting}' RETURN n.msg_reply AS reply")
        return results[0]['reply'] if results else None
    return None

def check_previous_question(question):
    result = run_query('MATCH (q:Question {text: $question})-[:HAS_ANSWER]->(a:Answer) RETURN a.text AS answer', {"question": question})
    return result[0]['answer'] if result else None

def is_similar_query(user_query, expected_queries):
    user_vec = model.encode(user_query, convert_to_tensor=True, normalize_embeddings=True)
    return any(util.cos_sim(user_vec, model.encode(expected, convert_to_tensor=True, normalize_embeddings=True)) > 0.7 for expected in expected_queries)

def remove_endings(text):
    endings = ["ครับ", "ค่ะ", "น้ะ", "นะ", "นะจ้ะ"]
    for ending in endings:
        text = text.replace(ending, "")
    return text.strip()

def fetch_product_info(search_term):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')  # Run in headless mode if desired

    with webdriver.Chrome(options=chrome_options) as driver:
        driver.get(f"https://www.bakeryclick.com/search?q={search_term}")
        driver.implicitly_wait(10)
        html = driver.page_source

    results = []
    mysoup = BeautifulSoup(html, "html.parser")
    
    product_cards = mysoup.find_all("div", class_="product_name")

    for card in product_cards:
        # Extract product title
        title_text = card.get_text(strip=True)

        # Extract product link
        link_element = card.find("a")
        link = f"https://www.bakeryclick.com{link_element['href']}" if link_element else "Link not available"

        # Extract additional product data from the gaeepd attribute
        gaeepd_data = link_element.get('gaeepd')
        product_info = json.loads(gaeepd_data.replace('&quot;', '"')) if gaeepd_data else {}

        # Extract product price
        price_text = product_info.get("price", "Price not available")
        price_text = f"{price_text} บาท" if price_text != "Price not available" else price_text

        # Append the product details to the results list
        results.append({
            'title': title_text,
            'price': price_text,
            'link': link
        })

    # Return a maximum of 5 results
    return results[:5] if results else None



# Flask app
app = Flask(__name__)
with open('usr_champ.txt', 'r') as file:
    channel_access_token, channel_secret = [line.strip() for line in file.readlines()]

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        line_bot_api = LineBotApi(channel_access_token)
        handler = WebhookHandler(channel_secret)
        handler.handle(body, request.headers['X-Line-Signature'])

        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        uid = json_data['events'][0]['source']['userId']
        global search_term 
        global price_min
        msg = remove_endings(msg)
        global is_lower_selected
        
        if "เมนู" in msg:
            quick_reply_options = [
                QuickReplyButton(action=MessageAction(label="ไม้นวดแป้ง", text="ค้นหา ไม้นวดแป้ง")),
                QuickReplyButton(action=MessageAction(label="แป้งทำขนม", text="ค้นหา แป้งทำขนม")),
                QuickReplyButton(action=MessageAction(label="กล่อง", text="ค้นหา กล่อง")),
            ]
            quick_reply = QuickReply(items=quick_reply_options)
            line_bot_api.reply_message(tk, TextSendMessage(text="ลูกค้าสนใจสินค้าแบบไหน:", quick_reply=quick_reply))

        if "ค้นหา" in msg:
            search_term = msg.replace("ค้นหา", "").strip()
            reply_text = "ลูกค้ามีงบประมาณราคาไม่เกินเท่าไหร่ครับ?\nหรือจะเลือกดูทั้งหมด(show all)ก็ได้ครับ"
            
            quick_reply_options = [
                QuickReplyButton(action=MessageAction(label="All", text="All")),
            ]
            quick_reply = QuickReply(items=quick_reply_options)
            
            line_bot_api.reply_message(tk, TextSendMessage(text=reply_text, quick_reply=quick_reply))

        if "ไม่เกิน" in msg:
            msg = msg.replace("ไม่เกิน", "").replace("ประมาณ", "").strip()
            price_min = re.findall(r'\d+', msg)
            price_min = ''.join(price_min)  
            is_lower_selected = True  

            product_info = fetch_product_info(search_term)
            if product_info is not None:
                def clean_price(price_str):
                    price_str = price_str.replace('฿', '').replace('บาท', '').replace(',', '').strip()
                    
                    # ตรวจสอบ
                    if '-' in price_str:
                        # แยกราคา
                        price_range = price_str.split('-')
                        min_price = price_range[0].strip()
                        return float(min_price)
                    else:
                        # ถ้าไม่มีช่วง ก็คืนราคาเดียว
                        return float(price_str)

                response_msg = (
                    "\n\n".join(
                        [
                            f"• ชื่อสินค้า: {item['title']}\n  ราคา: {item['price']}\n  ลิงค์: {item['link']}\n"
                            for item in product_info 
                            if item['price'] != "Price not available" and clean_price(item['price']) < int(price_min)
                        ]
                    ) if product_info else "ไม่พบสินค้าที่ท่านต้องการ"
                )

                if response_msg:
                    quick_reply_options = [
                        QuickReplyButton(action=MessageAction(label="All", text="All")),
                    ]
                    quick_reply = QuickReply(items=quick_reply_options)

                    line_bot_api.reply_message(tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))
                else:
                    line_bot_api.reply_message(tk, TextSendMessage(text="ขออภัย ไม่มีรายการสินค้าที่ตรงกับความต้องการของคุณ"))
            elif product_info == None:
                line_bot_api.reply_message(tk, TextSendMessage(text="ขออภัย ไม่มีรายการสินค้าที่ตรงกับความต้องการของคุณ"))
            

        if "All" in msg:
            product_info = fetch_product_info(search_term)
            is_lower_selected = False 

            if product_info is not None:
                response_msg = (
                    "\n\n".join(
                        [
                            f"• ชื่อสินค้า: {item['title']}\n  ราคา: {item['price']}\n  ลิงค์: {item['link']}\n"
                            for item in product_info 
                            if item['price'] != "Price not available"
                        ]
                    ) if product_info else "ไม่พบข้อมูลสินค้า"
                )

                if response_msg:
                    line_bot_api.reply_message(tk, TextSendMessage(text=response_msg))
                else:
                    line_bot_api.reply_message(tk, TextSendMessage(text="ขออภัย ไม่มีรายการสินค้าที่ตรงกับความต้องการของคุณครับ"))
            elif product_info == None:
                line_bot_api.reply_message(tk, TextSendMessage(text="ขออภัย ไม่มีรายการสินค้าที่ตรงกับความต้องการของคุณครับ"))

        # name input
        if "ชื่อ" in msg and "อะไร" in msg:
            user_name = get_user_name(uid)
            if user_name:
                line_bot_api.reply_message(tk, TextSendMessage(text=f"ชื่อของคุณคือ {user_name} ค่ะ"))
            else:
                line_bot_api.reply_message(tk, TextSendMessage(text="ขอโทษค่ะ ฉันไม่ทราบชื่อของคุณ"))

        elif "ชื่อ" in msg and "เชื่อ" not in msg:
            name = msg.split("ชื่อ")[-1].strip()
            if name:
                save_user_info(uid, name)
                line_bot_api.reply_message(tk, TextSendMessage(text=f"ขอบคุณที่แนะนำตัวค่ะ {name}"))
            else:
                line_bot_api.reply_message(tk, TextSendMessage(text="ไม่สามารถระบุชื่อได้ กรุณาระบุชื่อของคุณค่ะ"))

        user_name = get_user_name(uid)
        if user_name and is_similar_query(msg, ["ชื่ออะไร", "ผมชื่ออะไร", "ชื่อของฉัน"]):
            line_bot_api.reply_message(tk, TextSendMessage(text=f"ชื่อของคุณคือ {user_name} ค่ะ"))

        response_msg = compute_response(msg)

        if response_msg:
            line_bot_api.reply_message(tk, TextSendMessage(text=response_msg + " ค่ะ"))
            log_chat_history(uid, msg, response_msg) 
        else:
            previous_answer = check_previous_question(msg)
            if previous_answer:
                line_bot_api.reply_message(tk, TextSendMessage(text=previous_answer + " ค่ะ"))
            else:
                print(response_msg)
                payload = {
                    "model": "supachai/llama-3-typhoon-v1.5",
                    "prompt": f"ผู้ตอบเป็นผู้เชี่ยวชาญเรื่องเบเกอรี่ ผู้ถามชื่อ คุณ{user_name} ตอบสั้นๆไม่เกิน 20 คำ เกี่ยวกับ '{msg}'",
                    "stream": False
                }
                response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
                if response.status_code == 200:
                    decoded_text = response.json().get("response", "")
                    line_bot_api.reply_message(tk, TextSendMessage(text=decoded_text + 'ครับ'))
                    save_response(uid, msg, decoded_text)  # Save the answer and response
                else:
                    print(f"Failed to get a response from Ollama: {response.status_code}, {response.text}")
                    line_bot_api.reply_message(tk, TextSendMessage(text="เกิดข้อผิดพลาดในการติดต่อ LLaMA"))

    except InvalidSignatureError:
        return jsonify({'message': 'Invalid signature!'}), 400

    return jsonify({'status': 'OK'}), 200

if __name__ == "__main__":
    app.run(port=5000)
