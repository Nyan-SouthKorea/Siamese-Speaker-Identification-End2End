import pyaudio
import wave
import sys

def record_audio(sample_rate, chunk_sec, channels, format, name_cnt):
    p = pyaudio.PyAudio()

    print("녹음을 시작합니다. 엔터를 누르세요...")
    input()  # 사용자가 엔터를 누를 때까지 대기합니다.
    print(f"{chunk_sec}초 동안 녹음을 시작합니다.")

    # chunk 단위로 데이터를 처리할 때 샘플의 수를 계산합니다.
    chunk = int(sample_rate * chunk_sec)
    stream = p.open(format=format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)


    frames = []
    for i in range(0, int(sample_rate / chunk * chunk_sec)):
        data = stream.read(chunk)
        frames.append(data)

        # 각 청크 데이터를 파일로 저장합니다.
        file_name = f'recorded_{name_cnt}.wav'
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(data)
        wf.close()

        print(f"{file_name} 저장됨")

    print("녹음이 완료되었습니다.")

    # 스트림을 정리합니다.
    stream.stop_stream()
    stream.close()
    p.terminate()

name_cnt = 0
while True:
    record_audio(sample_rate=16000, chunk_sec=3, channels=1, format=pyaudio.paInt16, name_cnt = name_cnt)
    name_cnt += 1

