FROM ubuntu:24.04
ENV PADDLEOCR_RELEASE=PaddleOCR-json_v1.4.1_debian_x64_glibc2.31
RUN apt-get update && apt-get install -y wget xz-utils
RUN cd /opt && wget https://github.com/hiroi-sora/PaddleOCR-json/releases/download/v1.4.1/${RELEASE}.tar.xz && tar -xvf ${RELEASE}.tar.xz
RUN cd /opt && touch 123
WORKDIR /opt/PaddleOCR-json_v1.4.1_debian_x64_glibc2.31
CMD ["./run.sh", "-use_gpu", "-addr", "any", "-port", "2225"]