FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install nano wget curl git unzip sudo golang
RUN DEBIAN_FRONTEND=noninteractive  apt install -y  --no-install-recommends golang sudo tesseract-ocr libtesseract-dev
RUN cd ~/ && go mod init test && go get -u -d gocv.io/x/gocv
RUN cd /root/go/pkg/mod/gocv.io/x/gocv@v0.42.0 && sed -i 's/WITH_GSTREAMER=ON/WITH_GSTREAMER=ON -D CUDNN_INCLUDE_DIR=\/usr\/include\/x86_64-linux-gnu -D CUDNN_LIBRARY=\/usr\/lib\/x86_64-linux-gnu\/libcudnn.so/g' Makefile && \
    make install_cuda
#RUN #go get -u -d gocv.io/x/gocv && cd ~/go/src/gocv.io/x/gocv && make install_cuda
##RUN pip install flask numpy imutils
#RUN cd /opt && mkdir opencv && cd opencv && git clone https://github.com/opencv/opencv.git --branch 4.12.0 && git clone https://github.com/opencv/opencv_contrib.git --branch 4.12.0 \
#    && cd opencv && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE       -D CMAKE_INSTALL_PREFIX=/usr/local       -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules       -D WITH_CUDA=ON       -D WITH_CUDNN=ON       -D CUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu       -D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so       -D ENABLE_FAST_MATH=1       -D CUDA_FAST_MATH=1       -D WITH_CUBLAS=1      -D BUILD_EXAMPLES=OFF -D ENABLE_CXX11=ON -D WITH_INF_ENGINE=ON -D InferenceEngine_DIR=/usr/local/dldt/inference-engine/build -D BUILD_DOCS=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=ON -D WITH_TBB=ON -DBUILD_opencv_cudacodec=OFF -D OPENCV_DNN_CUDA=ON -D CUDA_GENERATION=Auto -DOPENCV_GENERATE_PKGCONFIG=ON -D BUILD_SHARED_LIBS=OFF .. && make -j 15
## RUN cp /opt/opencv/opencv/build/lib/python3/cv2.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/dist-packages/ \
##    ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/opencv4/3rdparty:${LD_LIBRARY_PATH}
##export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
##export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
WORKDIR /var/www/go-lineage2-server
ENV PADDLEOCR_RELEASE=PaddleOCR-json_v1.4.1_debian_x64_glibc2.31
RUN mkdir /opt/paddleocr
COPY ${PADDLEOCR_RELEASE} /opt/paddleocr
CMD ["go", "run", "main.go", "web-server"]