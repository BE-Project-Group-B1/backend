import speech_recognition as sr
import os
from pydub import AudioSegment
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import SepformerSeparation as separator
from transformers import pipeline
import torchaudio
class AudioTools:
    def __init__(self):
        self.verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
        self.separator = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')
        self.sr = sr.Recognizer()
        self.summarizer=pipeline("summarization", model="facebook/bart-large-cnn")

    def transcript(self,filePath="received.wav"):   

        with sr.AudioFile(filePath) as source:
                audio = self.sr.record(source)  # read the entire audio file                  
                return self.sr.recognize_google(audio)

    def identify(self,hash):
        score, prediction = self.verification.verify_files("received.wav", "{}.wav".format(hash))
        return prediction[0]
    def separate(self):
        est_sources = self.separator.separate_file(path='received.wav')
        res=[]
        c=1
        while(True):
            try:
                file=f'amazing_sounds.wav'
                torchaudio.save(file, est_sources[:,:,c], 8000)
                c+=1
                t=self.transcript(file)
                res.append(t)
            except Exception as e:
                print(e)
                break  
        print(res)
        return res
    def summarize(self,text):
        maxl=len(text.split())//1.5
        summ=self.summarizer(text, max_length=60, min_length=30, do_sample=False)[0]['summary_text']
        return summ
