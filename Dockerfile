FROM ubuntu:latest
ADD requirements.txt .
ADD detect_birds.py . 
ADD bird_data.csv .

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    portaudio19-dev

RUN pip3 install -r requirements.txt

CMD ["python3", "./detect_birds.py"]