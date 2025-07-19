FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu20.04
RUN apt-get update && apt-get -y install nano wget curl git unzip sudo python3 python3-pip
RUN DEBIAN_FRONTEND=noninteractive  apt install -y  --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev
RUN pip install flask numpy imutils
RUN cd /opt && mkdir opencv && cd opencv && git clone https://github.com/opencv/opencv.git --branch 4.12.0 && git clone https://github.com/opencv/opencv_contrib.git --branch 4.12.0 \
    && cd opencv && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE       -D CMAKE_INSTALL_PREFIX=/usr/local       -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules       -D WITH_CUDA=ON       -D WITH_CUDNN=ON       -D CUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu       -D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so       -D ENABLE_FAST_MATH=1       -D CUDA_FAST_MATH=1       -D WITH_CUBLAS=1       -D BUILD_opencv_python3=ON       -D BUILD_EXAMPLES=OFF .. && make install
 RUN cp /opt/opencv/opencv/build/lib/python3/cv2.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/dist-packages/