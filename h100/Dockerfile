# longtou.2024
FROM gcr.io/deeplearning-platform-release/pytorch-cu121.2-3.py310

RUN apt update && apt install -y file # webdataset cache_dir error
WORKDIR /root
ADD CosyVoice /root/CosyVoice
WORKDIR /root/CosyVoice
RUN ./install_espeak.sh
RUN ./setup_miniforge.sh conda cosyvoice 3.10
RUN  . ./activate_python.sh  && pip install -r requirements.txt
WORKDIR /root/CosyVoice/examples/aihub/cosyvoice2

CMD ["/bin/bash"]
