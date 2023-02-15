import json
import torch
import sys
import os
dirpath = os.path.dirname(__file__)
sys.path.append(dirpath)

import numpy as np
import commons
import utils
# from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io import wavfile

import pyaudio
import wave
import soundfile
# from ao import AudioDevice

###########################
#### Global Configs
############################

TEMP_WAVE_FILE = "temp_folder/temp_final_out.wav"
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
# RATE = 16000

class TTS_Model:
    def __init__(self) -> None:
        with open(os.path.join(dirpath, 'configs/mode.json'), 'r') as f:
            json_elements = json.load(f)
            self.chinese_mode = json_elements['chinese_mode']
            self.sampling_rate = json_elements['sampling_rate']
            self.noise_scale = json_elements['noise_scale'] 
            self.noise_scale_w = json_elements['noise_scale_w']
            
        if self.chinese_mode:
            self.net_g, self.hps = self.__init_chinese()
        else:
            self.net_g, self.hps = self.__init_english()
        
        # player
        self.audio = pyaudio.PyAudio()
    
    def __init_chinese(self):
        hps = utils.get_hparams_from_file(os.path.join(dirpath, "configs/woman_csmsc.json"))

        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model).cuda()
        _ = net_g.eval()
        
        _ = utils.load_checkpoint(os.path.join(dirpath, "models/pretrained_ljs_zh.pth"), net_g, None) 
        
        return net_g, hps

    def __init_english(self):
        hps = utils.get_hparams_from_file(os.path.join(dirpath, "configs/ljs_base.json"))

        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model).cuda()
        _ = net_g.eval()
        
        _ = utils.load_checkpoint(os.path.join(dirpath, "models/pretrained_ljs.pth"), net_g, None) 
        
        return net_g, hps
    
        
    # def __readRawframes(self, nframes):
    #     if self._data_seek_needed:
    #         self._data_chunk.seek(0, 0)
    #         pos = self._soundpos * self._framesize
    #         if pos:
    #             self._data_chunk.seek(pos, 0)
    #         self._data_seek_needed = 0
    #     if nframes == 0:
    #         return b''
    #     data = self._data_chunk.read(nframes * self._framesize)
    #     if SAMP != 1 and sys.byteorder == 'big':
    #         data = audioop.byteswap(data, self._sampwidth)
    #     if self._convert and data:
    #         data = self._convert(data)
    #     self._soundpos = self._soundpos + len(data) // (self._nchannels * self._sampwidth)
    #     return data
            
    def get_text(self, text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def preprocess(self, text : str):
        if self.chinese_mode:
            return text.strip().replace(',', '，').replace('.', '。').replace('?','？').replace("!","！")
        else:
            return text.strip()
    
    def text2speech(self, text) -> np.ndarray: 
        # text = self.preprocess(text)
        stn_tst = self.get_text(text, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            audio = self.net_g.infer(x_tst, x_tst_lengths, noise_scale=self.noise_scale, noise_scale_w=self.noise_scale_w, length_scale=1)[0][0,0].data.cpu().float().numpy()
        return audio

    ##
    # TODO : FIX This speech function 
    def text2speechAndPlay(self, text):
        raw_audio = self.text2speech(text)
        # ipd.display(ipd.Audio(audio, rate=self.sampling_rate, normalize=False)) #self.hps.data.sampling_rate
        # print(len(audio))        
        # os.makedirs(os.path.dirname(TEMP_WAVE_FILE), exist_ok=True)
        # wavfile.write(TEMP_WAVE_FILE, self.sampling_rate, audio)
        soundfile.write(TEMP_WAVE_FILE, raw_audio, self.sampling_rate, subtype='PCM_16')
        #

        # play
        with wave.open(TEMP_WAVE_FILE, 'rb') as wf:
            p = pyaudio.PyAudio()
            stream=p.open(
                format=pyaudio.paInt16, channels=CHANNELS, rate=self.sampling_rate, output=True)
        
            while len(data := wf.readframes(CHUNK)):  
                stream.write(data)
                
            stream.close()
            p.terminate()

        
    def text2wav(self, text, filename=TEMP_WAVE_FILE):
        audio = self.text2speech(text)
        os.makedirs(os.path.dirname(TEMP_WAVE_FILE), exist_ok=True)
        wavfile.write(filename, self.sampling_rate, audio.astype(np.int16))
        

if __name__ == "__main__":
    model = TTS_Model()
    model.text2speechAndPlay("你好，我是工具人") 