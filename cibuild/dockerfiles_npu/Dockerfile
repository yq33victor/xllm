FROM jd-train-cn-north-1-inner.jcr.service.jdcloud.com/9n-triton-dev:centos7.9.2009  

# 有些镜像没有正确配置sshd，建议增加这一行防止意外情况
RUN echo >> /etc/ssh/sshd_config && echo "Port 22" >> /etc/ssh/sshd_config && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && ssh-keygen -A

# Install miniconda
RUN sed -e 's|^mirrorlist=|#mirrorlist=|g' \
    -e 's|^#baseurl=http://mirror.centos.org/centos|baseurl=http://mirrors.jd.com/centos|g'\
    -i.bak /etc/yum.repos.d/Centos-Base.repo  && \
    yum install -y wget && \
    wget -O /etc/yum.repos.d/epel.repo http://mirrors.aliyun.com/repo/epel-7.repo && \
    yum makecache && \
    yum groupinstall -y "Development Tools" && \
    yum groupinstall -y "Development Tools" && \
    yum install -y \
      lftp \
      ccache \
      wget \
      psmisc \
      iptraf \
      gdb \
      lrzsz \
      which \
      bzip2 \
      net-tools \
      rsync \
      lsof \
      sysvinit-tools \
      zip \
      nscd \
      sysstat \
      ntp \
      ninja-build \
      vim \
      bison \
      msgpack \
      rust \
      cargo \
      perl-IPC-Cmd \
      numactl-devel \
      gmp-devel \
      zlib-devel \
      bzip2-devel \
      libffi-devel \
      openssl-devel \
      ncurses-devel \
      sqlite-devel \
      readline-devel \
      tk-devel \
      python3 \
      libarchive-devel \
      re2-devel \
      xz-devel \
      glibc-devel.i686 \
      perl-ExtUtils-MakeMaker \
      rapidjson-devel \
      texinfo \
      mlocate \
      bc \
      gdbm-devel \
      db4-devel \
      libpcap-devel \
      hdf5-devel \
      libicu \
      patch \
      pciutils \
      mesa-libGL \
      blas-devel \
      lapack-devel \
      python-backports-lzma \
      openblas.x86_64 \
      openblas-devel.x86_64 \
      git \
      gcc \
      gcc-c++ && \
    yum makecache && \
    yum clean all

# Install miniconda

RUN wget  https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/ascend-llm/mindie/1030poc/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash ./miniconda.sh -b -p /usr/local/miniconda3 && \
    rm -f miniconda.sh

ENV PATH=/usr/local/miniconda3/bin:$PATH
ENV ANACONDA_HOME=/usr/local/miniconda3

# Install gcc
RUN cd /tmp \
    && wget https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/ascend-llm/mindie/common/gcc-11.4.0.tar.gz \
    && tar zxf gcc-11.4.0.tar.gz \
    && rm -rf gcc-11.4.0.tar.gz \
    && cd /tmp/gcc-11.4.0 \
    && sed -i "s#base_url='ftp://gcc.gnu.org/pub/gcc/infrastructure/'#base_url='https://gcc.gnu.org/pub/gcc/infrastructure/'#" ./contrib/download_prerequisites \
    && sed -i "s#fetch='wget'#fetch='wget --no-check-certificate'#" ./contrib/download_prerequisites \
    && ./contrib/download_prerequisites \
    && mkdir build


RUN cd /tmp/gcc-11.4.0/build \
    && /tmp/gcc-11.4.0/configure --prefix=/usr --mandir=/usr/share/man --infodir=/usr/share/info --with-bugurl=http://bugzillla.redhat.com/bugzilla \
       --enable-bootstrap --enable-shared --enable-threads=posix --enable-checking=release --with-system-zlib --enable-___cxa_atexit \
       --disable-libunwind-exceptions --enable-gnu-unique-object --enable-linker-build-id --with-linker-hash-style=gnu \
       --enable-lannguages=c,c++,objc,obj-c++,fortran,go,lto --enable-plugin --enable-initfini-array --disable-libgcj \
       --with-isl --with-cloog --enable-gnu-iindirect-function --with-tune=generic --with-arch_32=x86-64 --build=x86_64-redhat-linux --disable-multilib \
    && make -j8 && make install \
    && rm -rf /tmp/gcc-11.4.0

# ENV TZ="Asia/Shanghai"


RUN echo >> /etc/ssh/sshd_config && echo "Port 22" >> /etc/ssh/sshd_config && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && ssh-keygen -A

WORKDIR /tmp
RUN cd /opt && \
    wget https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/ascend-llm/mindie/common/make-4.3.tar.gz && \
    tar axf make-4.3.tar.gz && \
    cd make-4.3 && \
    mkdir build && \
    cd build && \
    ../configure --prefix=/usr &&\ 
    make && \
    make install 

RUN cd /opt && \
    wget https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/ascend-llm/mindie/common/glibc-2.34.tar.gz && \
    tar -zxvf glibc-2.34.tar.gz && \
    cd glibc-2.34/ && \
    mkdir build && \
    cd build && \
    yum -y install bison && \
    export LD_LIBRARY_PATH= && \
    ../configure --prefix=/usr --disable-profile --enable-add-ons --with-headers=/usr/include --with-binutils=/usr/bin --disable-sanity-checks --disable-werror && \
    make -j128 && \
    LD_PRELOAD=/lib64/libc-2.17.so sln /opt/glibc-2.34/build/libc.so.6 /lib64/libc.so.6 && \
    LD_PRELOAD=/lib64/libc-2.17.so sln /opt/glibc-2.34/build/dlfcn/libdl.so.2 /lib64/libdl.so.2 && \
    LD_PRELOAD=/lib64/libc-2.17.so sln /opt/glibc-2.34/build/nptl/libpthread.so.0  /lib64/libpthread.so.0 && \
    LD_PRELOAD=/lib64/libc-2.17.so sln /opt/glibc-2.34/build/elf/ld-linux-x86-64.so.2 /usr/lib64/ld-linux-x86-64.so.2 && \
    make install || true && \
    rm -f /var/lib/rpm/__db.00* && \
    rpm --rebuilddb

RUN wget https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/ascend-llm/mindie/1030poc/cann/Ascend-cann-toolkit_8.0.T60_linux-x86_64.run && \
    wget https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/ascend-llm/mindie/1030poc/cann/Ascend-cann-kernels-910b_8.0.T60_linux-x86_64.run && \
    wget https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/ascend-llm/mindie/1030poc/cann/Ascend-cann-nnal_8.0.T60_linux-x86_64.run

ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest 
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/fwkacllib/python/site-packages/:${ASCEND_TOOLKIT_HOME}/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${ASCEND_TOOLKIT_HOME}/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH 
ENV OPTION_EXEC_EXTERN_PLUGIN_PATH=${ASCEND_TOOLKIT_HOME}/fwkacllib/lib64/plugin/opskernel/libfe.so:${ASCEND_TOOLKIT_HOME}/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:${ASCEND_TOOLKIT_HOME}/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so 
ENV PATH=$PATH:${ASCEND_TOOLKIT_HOME}/fwkacllib/ccec_compiler/bin/:${ASCEND_TOOLKIT_HOME}/toolkit/tools/ide_daemon/bin/ 
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/fwkacllib/lib64/:${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64:$LD_LIBRARY_PATH 
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH 
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/x86_64-linux/bin:$PATH  
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}  
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp  
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit  
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME} 
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/devlib/x86_64/
ENV LD_LIBRARY_PATH=/usr/local/miniconda3/lib/:$LD_LIBRARY_PATH
#ENV PATH=/usr/local/miniconda3/bin:$PATH


RUN echo y | bash ./Ascend-cann-toolkit*.run --install-path=/usr/local/Ascend/ --install --quiet --install-for-all && \ 
    echo y | bash ./Ascend-cann-kernels-*.run  --install --install-for-all 
    
RUN echo y | bash ./Ascend-cann-nnal*.run  --install-path=/usr/local/NNAL/ --install && \
    rm -rf /tmp/* 
    
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH 


# 安装cmake
RUN wget https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/ascend-llm/mindie/common/cmake-3.28.1-linux-x86_64.sh && \
    mkdir /usr/local/cmake  && \
    bash cmake-3.28.1-linux-x86_64.sh  --skip-license --prefix=/usr/local/cmake && \
    echo "export PATH=/usr/local/cmake/bin:/$PATH " >> /etc/bashrc  &&\
    echo "export LD_LIBRARY_PATH=/usr/local/cmake/lib:/$LD_LIBRARY_PATH " >> /etc/bashrc 
 




ENV PATH=/usr/local/cmake/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cmake/lib:/$LD_LIBRARY_PATH



RUN pip3 install -U pip && \
    pip3 install decorator  numpy==1.26.4 pandas sympy pyyaml pathlib2 grpcio grpcio-tools protobuf scipy \
    requests attrs wheel Pillow setuptools==57.4.0 matplotlib opencv-python pandas lxml pytest xdoctest   -i https://mirrors.jd.com/pypi/web/simple 

#torch.whl
#torch_npu
RUN pip install pip==24.0   -i https://mirrors.jd.com/pypi/web/simple  && \
    wget -O torch-2.1.0+cpu-cp310-cp310-linux_x86_64.whl https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/ascend-llm/mindie/730poc/v1.0/torch/torch-2.1.0%2Bcpu-cp310-cp310-linux_x86_64.whl && \
    wget https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/ascend-llm/mindie/1030poc/torch/torch_npu-2.1.0.post8-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl && \
    pip3 install torch-* --force-reinstall   -i https://mirrors.jd.com/pypi/web/simple  && \
    pip3 install torch_npu-2.1.0.post8-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl  --force-reinstall --no-deps -i https://mirrors.jd.com/pypi/web/simple  && \
    git config --global http.sslVerify false && \
    rm -rf /tmp/*  && \
    rm -rf /root/.cache/pip 
    
#apex包编译安装 
RUN wget https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/ascend-llm/mindie/1030poc/torch/apex-0.1.dev20241101%2Bascend-cp310-cp310-linux_x86_64.whl && \
    ls && \
    pip3 install apex-0.1.dev20241101+ascend-cp310-cp310-linux_x86_64.whl  && \
    rm -rf /tmp/* 

RUN pip3 install torchvision==0.16.0 cffi pycocotools \
    easydict psutil absl-py kiwisolver cython cycler mpmath attr  -i  https://mirrors.jd.com/pypi/web/simple && \
    pip3 install loguru transformers==4.41.0 tiktoken transformers_stream_generator==0.0.4 einops \
    decorator attrs psutil absl-py cloudpickle ml-dtypes scipy tornado tiktoken loguru  einops sentencepiece accelerate  -i https://mirrors.jd.com/pypi/web/simple  && \
    rm -rf /tmp/* && \
    rm -rf /root/.cache/pip

RUN rm -f /usr/local/miniconda3/lib/libcurl.so.4

RUN export http_proxy=http://bamboo-proxy.jd.com:80 https_proxy=http://bamboo-proxy.jd.com:80 && \
    wget https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/git-2.7.4.tar.gz && \
    tar -xvf git-2.7.4.tar.gz && \
    yum install -y libcurl-devel && \
    cd git-2.7.4 && \
    make -j128 prefix=/usr && \
    make install prefix=/usr && \
    cd .. && \
    rm -rf git-2.7.4 && \
    rm -rf git-2.7.4.tar.gz

RUN export http_proxy=http://bamboo-proxy.jd.com:80 https_proxy=http://bamboo-proxy.jd.com:80 && \
    wget https://9n-das-tools.s3.cn-north-1.jdcloud-oss.com/9n-triton/git-2.7.4.tar.gz && \
    tar -xvf git-2.7.4.tar.gz && \
    yum install -y libcurl-devel && \
    cd git-2.7.4 && \
    make -j128 prefix=/usr && \
    make install prefix=/usr && \
    cd .. && \
    rm -rf git-2.7.4 && \
    rm -rf git-2.7.4.tar.gz

# boost
RUN export http_proxy=http://bamboo-proxy.jd.com:80 https_proxy=http://bamboo-proxy.jd.com:80 && \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.gz && \
    tar -xvf boost_1_80_0.tar.gz && \
    cp -r boost_1_80_0/boost /usr/include/boost && \
    rm -rf boost_1_80_0 && \
    rm -rf boost_1_80_0.tar.gz


RUN wget https://9n-das-tools.s3-internal.cn-north-1.jdcloud-oss.com/9n-triton/ascend-llm/mindie/1030poc/torch/libtorch_npu.tar.gz && \
    tar -zxvf libtorch_npu.tar.gz -C /usr/local/

RUN echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /etc/bashrc 
RUN echo "source /usr/local/NNAL/nnal/atb/set_env.sh" >> /etc/bashrc 

# RUN yum install -y ccache ninja-build bison rust cargo perl-IPC-Cmd perl-ExtUtils-MakeMaker gmp-devel curl-devel

RUN pip install \
    torch>=2.1.0 \
    fastapi==0.115.6 \
    huggingface-hub==0.26.3 \
    shortuuid==1.0.13 \
    safetensors==0.4.5 \
    numpy==1.26.4 \
    uvicorn==0.32.1 \
    starlette==0.41.3 \
    pydantic==2.10.3 \
    typing-extensions==4.12.2 \
    filelock==3.16.1 \
    sympy==1.13.3 \
    networkx==3.4.2 \
    jinja2==3.1.4 \
    fsspec==2024.10.0 \
    packaging==23.0 \
    pyyaml==6.0.2 \
    requests==2.29.0 \
    tqdm==4.65.0 \
    click==8.1.7 \
    h11==0.14.0 \
    annotated-types==0.7.0 \
    pydantic-core==2.27.1 \
    anyio==4.7.0 \
    MarkupSafe==3.0.2 \
    charset-normalizer==2.0.4 \
    idna==3.4 \
    urllib3==1.26.16 \
    certifi==2023.5.7 \
    mpmath==1.3.0 \
    exceptiongroup==1.2.2 \
    sniffio==1.3.1 \
    aiohttp == 3.10.11  \
    openai == 1.57.1  \
    transformers==4.43.1 \
    pillow == 11.0.0 \
    -i "https://mirrors.jd.com/pypi/web/simple"
