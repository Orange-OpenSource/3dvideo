# Pytorch official release only supports cuda 1..8 for now
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN echo "Installing apt packages..." \
	&& export DEBIAN_FRONTEND=noninteractive \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends \
	python3 \
	python3-pip \
    wget \
    imagemagick \
    git \
    libsqlite3-dev \
    python3-dev

# Compile the latest stable version of colmap binary and library because Ubuntu colmap package only installs
# the binary and without CUDA support and we need CUDA acceleration to speed up calibration
# Note: we cannot disable colmap UI with DGUI_ENABLED=OFF because SIFT requires OpenGL (see https://github.com/colmap/colmap/issues/1431)
# Regarding ceres-solver, we also use a specific version because LocalParametizations has been removed in latest versions
# and it is still used by Colmap
# Note that CMAKE_CUDA_ARCHITECTURES is set to 'all-major' to target all Marcel cluster GPUS as well as our local machines
RUN echo "Compiling colmap and dependencies" \
    && apt -y install libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev libboost-test-dev libeigen3-dev libsuitesparse-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libflann-dev \
	&& wget -q https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
	&& chmod u+x ./cmake-3.24.1-Linux-x86_64.sh \
	&& ./cmake-3.24.1-Linux-x86_64.sh --skip-license --prefix=/usr/local \
	&& rm ./cmake-3.24.1-Linux-x86_64.sh \
	&& git clone https://ceres-solver.googlesource.com/ceres-solver \
	&& cd ceres-solver \
	&& git checkout 2.1.0 \
	&& cmake -B build . -DCUDA=ON -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
	&& cmake --build build -j3 \
	&& cmake --install build \
	&& cd .. \
	&& git clone https://github.com/colmap/colmap.git \
	&& cd colmap \
	&& git checkout 2b1f843e23f4383c4e2667451ec73271fb67e96b \
	&& cmake -B build . -DTESTS_ENABLED=OFF -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES="all-major" \
	&& cmake --build build -j3 \
	&& cmake --install build \
	&& cd .. \
	&& rm -rf ceres-solver colmap

RUN echo "Installing pip packages..." \
	&& python3 -m pip install -U pip \
	&& pip3 --no-cache-dir install plyfile tqdm \
	&& pip3 --no-cache-dir install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

ENV TORCH_CUDA_ARCH_LIST "7.0 7.5 8.0 8.6 8.9+PTX"

RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
RUN pip3 install open3d tensorboard dacite
ENV PYTHONUNBUFFERED=1

# Copy only submodule packages to compile them.
# Note: If we copy here the whole repo, each change to the repo will lead to DockerFile
# next step execution (aka pytorch-based package recompilation) which is very long!
COPY ./submodules/simple-knn /gaussian-splatting-deps/simple-knn
RUN cd /gaussian-splatting-deps/simple-knn \
	&& python3 setup.py install

COPY ./submodules/diff-gaussian-rasterization /gaussian-splatting-deps/diff-gaussian-rasterization
RUN cd /gaussian-splatting-deps/diff-gaussian-rasterization\
	&& python3 setup.py install

# Now, python package are compiled, we can copy the whole repo
COPY ./ /gaussian-splatting

CMD  ["/bin/bash"]
