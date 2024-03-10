from tqdm import tqdm
from natsort import natsorted
import os
from IPython.display import Audio
import librosa
import copy
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
import wave
import shutil
import pyaudio
from collections import deque
import threading
import soundfile as sf
import torchaudio.transforms as T

path = 'D:/Code/240227_speaker_classification/custom'
database_path = f'{path}/database'
model_path = f'{path}/LSTMtoFCN_sigmoid-best.pt'
audio_len = 2 # 자를 오디오 길이
stream_save = False
n_mfcc = 40

if torch.cuda.is_available() == True:
    device = torch.device('cuda')
    print('현재 가상환경 cuda 설정 가능')
else:
    device = 'cpu'
    print('현재 가상환경 cpu 사용')

# listdir을 정렬하기
def listdir(path):
    return natsorted(os.listdir(path))

# 모델 선언
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=161, hidden_size=512, num_layers=1, batch_first=True) # mfcc shape의 마지막 차원 개수로 받음
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.1)
        
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=1024, num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm1d(1024)
        
        self.lstm3 = nn.LSTM(input_size=1024, hidden_size=2048, num_layers=1, batch_first=True)
        self.bn3 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.1)

        self.lstm4 = nn.LSTM(input_size=2048, hidden_size=2048, num_layers=1, batch_first=True)
        self.bn4 = nn.BatchNorm1d(2048)
        
        # 최종 출력을 위한 선형 레이어
        self.fc = nn.Linear(2048, 1024)  # n개의 

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        
        x, _ = self.lstm3(x)
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout2(x)

        x, _ = self.lstm4(x)
        x = self.bn4(x.transpose(1, 2)).transpose(1, 2)
        
        # 마지막 시퀀스의 출력만을 사용
        x = self.fc(x[:, -1, :])
        return x

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024*2, 512) # LSTM이 반환한 feature 개수 * 2
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid 활성화 함수 추가

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Sigmoid 적용
        return x

class LSTM_FCN(nn.Module):
    def __init__(self):
        super(LSTM_FCN, self).__init__()
        self.model_lstm = LSTM() # LSTM 모델 정의
        self.model_fcn = FCN() # FCN 모델 정의

    def forward(self, x1, x2):
        # LSTM을 이용해 feature 추출
        features1 = self.model_lstm(x1)
        features2 = self.model_lstm(x2)

        # 두 특징을 concatenate
        features = torch.cat((features1, features2), dim=1)

        # FCN 모델에 concatenate된 특징을 입력
        output = self.model_fcn(features)
        return output

# 데이터셋
class AudioDataset(Dataset):
    def __init__(self, audio_path, audio_len):
        self.audio_path = audio_path
        self.audio_len = audio_len
        # mfcc 관련
        self.mfcc_transform = T.MFCC(sample_rate=16000, n_mfcc=40) # melkwargs 설정하는 것 기본값으로

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        waveform1, sr1 = self.crop_and_pad(self.audio_path)
        waveform1 = torch.tensor(waveform1).float()
        mfcc1 = self.mfcc_transform(waveform1)  # MFCC 변환
        mfcc1 = (mfcc1 - mfcc1.mean()) / mfcc1.std() # 정규화(standardization)
        return mfcc1

    def crop_and_pad(self, audio_path):
        waveform, sr = sf.read(audio_path) # 오디오 파일 로드
        # 지정된 길이에 해당하는 샘플 수 계산
        target_length = int(self.audio_len * sr)
        # 현재 오디오 파일의 길이 계산
        current_length = waveform.shape[0]
        if current_length < target_length:
            # 지정된 길이 이하일 경우, 부족한 길이만큼 0으로 패딩
            padding = target_length - current_length
            waveform = np.pad(waveform, (0, padding), 'constant')
        else:
            # 지정된 길이 이상일 경우, 랜덤하게 해당 길이 구간을 선택하여 크롭
            start_point = random.randint(0, current_length - target_length)
            waveform = waveform[start_point:start_point + target_length]
        return waveform, sr

# 오디오 버퍼 관리
class buffer:
    def __init__(self, stream, chunk, rate, n_mfcc, stream_save):
        self.buffer = deque()
        self.stream = stream
        self.chunk = chunk
        self.rate = rate
        self.n_mfcc = n_mfcc
        self.cnt = 0
        self.stream_save = stream_save
        # 실시간 출력 설정
        self.pyaudio_instance = pyaudio.PyAudio()
        self.output_stream = self.pyaudio_instance.open(format=pyaudio.paInt16,channels=1,rate=self.rate,output=True)

    def stream_start(self):
        print('스트리밍 시작') 
        while True:  
            data = self.stream.read(self.chunk) # 2초의 데이터가 빠짐
            self.buffer.append(data) # deque 데이터에 추가
            self.output_stream.write(data)

    def popleft(self):
        buffer_list = self.buffer.popleft() # 선입선출 하나 제거
        # wave 모듈을 사용하여 오디오 파일로 저장
        with wave.open(f'{path}/tmp.wav', 'wb') as wf:
            wf.setnchannels(1)  # 모노 채널
            wf.setsampwidth(self.pyaudio_instance.get_sample_size(pyaudio.paInt16))  # 샘플 너비 설정
            wf.setframerate(self.rate)  # 샘플링 레이트 설정
            wf.writeframes(buffer_list)  # 오디오 데이터 쓰기


# 오디오 설정
format = pyaudio.paInt16 # 데이터 형식
channels = 1
rate = 16000 # 샘플링 레이트
chunk = rate * audio_len # 블록 크기
audio = pyaudio.PyAudio() # PyAudio 시작

# 스트리밍 시작
stream = audio.open(format=format, channels=channels,
                    rate=rate, input=True,
                    frames_per_buffer=chunk) 

buffer_class = buffer(stream, chunk, rate, n_mfcc, stream_save)
multithread = threading.Thread(target = buffer_class.stream_start)
multithread.start()

# 모델 불러오기
model = nn.DataParallel(LSTM_FCN()).to(device)
model.load_state_dict(torch.load(model_path))
model = model.module
model = model.to('cuda:0')

model.eval()  # 평가 모드로 설정
print('모델 불러오기 완료')

# 데이터베이스에 있는 음성들 feature 추출 후 저장
print('데이터베이스 feature 추출 중...')
database = {}
for audio_name in listdir(database_path):
    if not '.wav' in audio_name: continue
    dataset = AudioDataset(f'{database_path}/{audio_name}', audio_len)
    data_loader = torch.utils.data.DataLoader(dataset)
    for mfcc in data_loader:
        with torch.no_grad():
            feature1 = model.model_lstm(mfcc.to('cuda:0'))
    database[audio_name.split('.')[0]] = feature1
# torch.size([1, 1024]) 형태로 database에 저장됨


while True:
    if len(buffer_class.buffer) > 0:
        buffer_class.popleft() # 음성 불러오기
        # LSTM에서 feature 뽑아내기
        dataset = AudioDataset(f'{path}/tmp.wav', audio_len)
        data_loader = torch.utils.data.DataLoader(dataset)
        for mfcc in data_loader:
            with torch.no_grad():
                feature2 = model.model_lstm(mfcc.to('cuda:0'))
        # database에 있는 feature들과 concat하여 fcn 추론
        pred_dic = {}
        for user, feature1 in database.items():
            concat_data = torch.cat((feature1, feature2), dim=1).to('cuda:0')
            output = model.model_fcn(concat_data)
            pred_dic[user] = output.item()
        pred_dic = dict(sorted(pred_dic.items(), key=lambda item: item[1], reverse=True))

        # 일치도의 평균을 내기
        av_score = {}
        for user, conf in pred_dic.items():
            user = user.split('_')[0]
            if user in av_score:
                av_score[user].append(conf)
            else:
                av_score[user] = [conf]
        new_av_score = {}
        for user, conf_list in av_score.items():
            new_av_score[user] = round(sum(conf_list) / len(conf_list), 3)

        # 프린트
        for user, conf in new_av_score.items():
            if 'background' in user:
                break
            print(f'화자 인식: {user} | 순위표: {new_av_score}')
            break
            
            
        


    


    