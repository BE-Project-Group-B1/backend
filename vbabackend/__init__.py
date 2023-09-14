
from flask import Flask
from flask import request,jsonify
from flask_cors import CORS
import os
import io
import soundfile
from db import users
from audio_tools import audiotools
from utils import randomwords
from utils import hashing
db=users.Users()
at=audiotools.AudioTools()
cwd=os.getcwd()
os.chdir(cwd+"/audiodb/")
at.separate()
app = Flask(__name__)

CORS(app)
class Store:
    def store(self,val):
        self.val=val
curr=Store()

@app.post("/genWords")
def genWords():
    temp=randomwords.genWords(4)
    print(temp)
    curr.store(temp)
    response = jsonify(temp)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.post("/isuser")
def isuser():
    username=request.form['username']
    response=None
    if(db.isUser(username)):
        response = jsonify("TRUE") 
    else:
        response = jsonify("FALSE")
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.post("/register")
def register():
    files = request.files
    file=files.get('file')
    username=request.form['username']
    password=hashing.genhash()
    res=db.newUser(username,password)
    os.chdir(cwd+"/audiodb/")
    filepath = os.path.join("{}.wav".format(password))
    file.save(filepath)
    file.seek(0)
    data, samplerate = soundfile.read(file)
    with io.BytesIO() as fio:
        soundfile.write(
            fio, 
            data, 
            samplerate=samplerate, 
            subtype='PCM_16', 
            format='wav'
        )
        data = fio.getvalue()
    print('voice registered for ',username)
    response = jsonify("DONE")
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.post("/login")
def login():
    files = request.files
    file=files.get('file')
    username=request.form['username']
    # Write the data to a file.
    os.chdir(cwd+"/audiodb/")
    filepath = os.path.join("received.wav")
    file.save(filepath)

    # Jump back to the beginning of the file.
    file.seek(0)
    # Read the audio data again.
    data, samplerate = soundfile.read(file)
    with io.BytesIO() as fio:
        soundfile.write(
            fio, 
            data, 
            samplerate=samplerate, 
            subtype='PCM_16', 
            format='wav'
        )
        data = fio.getvalue()
    hash=db.getUserHash(username)
    recd=at.transcript()
    result=at.identify(hash)
    print("IDENTIFICATION:",result)
    response=None
    print("GEN:",curr.val,"\n","RECD:",recd)
    if(result and curr.val.lower()==recd.lower()):
        print('Access granted to ',username)
        response = jsonify("SUCCESS")
    else:
        response = jsonify("FAIL")
    
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
@app.post("/transcript")
def transcript():
    files = request.files
    file=files.get('file')
    os.chdir(cwd+"/audiodb/")
    filepath = os.path.join("received.wav")
    file.save(filepath)

    # Jump back to the beginning of the file.
    file.seek(0)
    # Read the audio data again.
    data, samplerate = soundfile.read(file)
    with io.BytesIO() as fio:
        soundfile.write(
            fio, 
            data, 
            samplerate=samplerate, 
            format='wav'
        )
        data = fio.getvalue()
    recd=at.transcript()
    response = jsonify(recd)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def main():
	app.run()
