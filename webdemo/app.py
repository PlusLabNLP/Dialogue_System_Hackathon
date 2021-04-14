from flask import Flask, render_template, request, jsonify, redirect, url_for, json
from flask import send_from_directory
import os
from apis.DialogSys import DialogSys
import pdb
#these are for sql
#import requests
#from sqlalchemy.orm import sessionmaker, scoped_session
#from sqlalchemy import create_engine
#from sql_model import dialog
#some_engine = create_engine('mysql+mysqlconnector://zixi:200683088@localhost:3308/dialogsys')
#session_factory = sessionmaker(bind=some_engine)
#Session = scoped_session(session_factory)
#session_tmp = Session()
#----------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret' 
chatbot = DialogSys()

@app.route('/', methods=['GET'])
def main(): 
  if request.method =='GET': 
    print(app.static_folder)
    return render_template('page.html')

@app.route('/generate/', methods=['POST'])
def generate():
  if request.method=='POST':
    data = json.loads(request.data)
    print('data', data)
    if data['convline-onoff']=='on':
        response = chatbot.decode(data)
        #response = {'responses':[['sys_utt'], ['None']], 'tokenized_message_list': 'message_list'+'new_message', 'keywords': 'A # B # C'}
        #pdb.set_trace()
        print('data', data)
        print('response', response)
    else:
        response = chatbot.baseline(data)
        print('data', data)
        print('response', response)

    return json.dumps(response)

@app.route('/regenerate-keywords/', methods=['POST'])
def regenerate_keywords():
    data = json.loads(request.data)
    print('data', data)
    keywords = chatbot.regenerate_keywords(data)
    print(keywords)
    return keywords

@app.route('/single-rerun/', methods=['POST'])
def single_rerun():
  print('single-rerun')
  if request.method=='POST':
    data = json.loads(request.data)
    response = chatbot.customize_decode(data)
  return response

@app.route('/database/insert', methods=['POST'])
def save_to_db():
  if request.method=='POST':
    #pdb.set_trace()
    #session = Session()
    #data = json.loads(request.data)
    #new_dialog = dialog(user_id=data['user_id'], log=data['log'])
    #session.add(new_dialog)
    #session.commit()
    #session.close()
    pass
    return 'succeed'


@app.route('/favicon.ico')
def favicon():
  print('send favicon file')
  return send_from_directory(os.path.join(app.root_path, 'static/img'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
  #return app.send_static_file('static/img/favicon.ico')
  


if __name__ == '__main__':
  # app.run(host='0.0.0.0', port=5000, use_debugger=False, use_reloader=False, passthrough_errors=True)
  app.run(host='0.0.0.0', port=12095, debug=False)
