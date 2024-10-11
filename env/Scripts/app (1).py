from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from sentence_transformers import SentenceTransformer, util,InputExample
from sentence_transformers import models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
# from ollama import Ollama  # Adjust the submodule name based on your findings
import numpy as np
from pydantic import BaseModel
import requests


model = SentenceTransformer('sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens')
import json
# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "test")

def get_llama_response(prompt):
   OLLAMA_API_URL = "http://localhost:11434/api/generate"
   headers = {
      "Content-Type": "application/json"
   }
   
   role_prompt = f"""
   ผู้ตอบเป็น Barista ที่มีความชำนาญในการชงกาแฟ และรอบรู้ด้านต่างๆเกี่ยวกับอุปกรณ์ และวัตถุดิบ {prompt} 
   โดยคำตอบยาวไม่เกิน 20 คำ
   """

   
   payload = {
      "model": "supachai/llama-3-typhoon-v1.5",
      "prompt": role_prompt,
      "stream": False
   }
   
   response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
   
   if response.status_code == 200:
      response_data = response.text
      data = json.loads(response_data)
      return data.get("response", "ขอโทษด้วย ฉันไม่สามารถให้คำตอบนี้ได้")  # Default message if response not found
   else:
      print(f"Failed to get a response: {response.status_code}, {response.text}")
      return "ขอโทษด้วย ฉันไม่สามารถให้คำตอบนี้ได้"


def run_query(query, parameters=None):
   with GraphDatabase.driver(URI, auth=AUTH) as driver:
       driver.verify_connectivity()
       with driver.session() as session:
           result = session.run(query, parameters)
           return [record for record in result]
   driver.close()


cypher_query = '''
MATCH (n:Barista) RETURN n.question as question, n.msg_reply as reply;
'''
greeting_corpus = []
greeting_vec = None
results = run_query(cypher_query)
for record in results:
   greeting_corpus.append(record['question'])
greeting_corpus = list(set(greeting_corpus))
print(greeting_corpus)  

def compute_similar(corpus, sentence):
   a_vec = model.encode([corpus],convert_to_tensor=True,normalize_embeddings=True)
   b_vec = model.encode([sentence],convert_to_tensor=True,normalize_embeddings=True)
   similarities = util.cos_sim(a_vec, b_vec)
   return similarities


def neo4j_search(neo_query):
   results = run_query(neo_query)
   # Print results
   for record in results:
       response_msg = record['reply']
   return response_msg     

def create_barista_node(question, reply):
    create_query = f'''
    CREATE (:Barista {{question: '{question}', msg_reply: '{reply}'}})
    '''
    run_query(create_query)

def compute_response(sentence):
   #print(greeting_corpus)
   greeting_vec = model.encode(greeting_corpus, convert_to_tensor=True,normalize_embeddings=True)
   ask_vec = model.encode(sentence, convert_to_tensor=True,normalize_embeddings=True)
   #Compute cosine-similarities
   greeting_scores = util.cos_sim(greeting_vec, ask_vec) 
   greeting_score = greeting_scores.cpu()
   greeting_np = greeting_score.numpy()
   # Extract the maximum similarity score as a scalar
   max_greeting_score = np.argmax(greeting_np)
   #greeting_np = np.array(greeting_scores)
   Match_Question = greeting_corpus[max_greeting_score] 
   if greeting_np[np.argmax(greeting_np)] > 0.7 :
        My_cypher = f"MATCH (n:Barista) where n.question ='{Match_Question}' RETURN n.msg_reply as reply"
        my_msg  = neo4j_search(My_cypher)
   else:
        # If no similar question is found, query Ollama
        my_msg = get_llama_response(sentence)  
        create_barista_node(sentence,my_msg)
   print(my_msg)
   return my_msg   
app = Flask(__name__)




@app.route("/", methods=['POST'])
def linebot():
   body = request.get_data(as_text=True)                   
   try:
       json_data = json.loads(body)                       
       access_token = 'Nh8mjNfdPX4G9D5DR3a6vuCWy/tZ8qql7NNocTnTreCMktobm+ju5Y/5UaFbx+nFyJeKTRm9twYWQOxNgVG5mpEr9EtqLc0YZMCj9MyFMymaSe+fhvMiUe4efdcRrE8XYCr9kbDN0VGzLHcsH1WT0gdB04t89/1O/w1cDnyilFU='
       secret = '0101124cb552df552fb5c81e8dbe76e7'
       line_bot_api = LineBotApi(access_token)             
       handler = WebhookHandler(secret)                   
       signature = request.headers['X-Line-Signature']     
       handler.handle(body, signature)                     
       msg = json_data['events'][0]['message']['text']     
       tk = json_data['events'][0]['replyToken']           
       response_msg = compute_response(msg)
       line_bot_api.reply_message( tk, TextSendMessage(text=response_msg) )
       print(msg, tk)                                     
   except:
       print(body)                                         
   return 'OK'               
if __name__ == '__main__':
   app.run(port=5000)
