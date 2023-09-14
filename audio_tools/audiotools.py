import speech_recognition as sr
import os
from pydub import AudioSegment
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
class AudioTools:
    def __init__(self):
        self.verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
        self.separator = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')
        self.sr = sr.Recognizer()

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