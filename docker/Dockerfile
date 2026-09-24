# CogNav: ROS 2 Humble, the workspace and two conda environments.
#
#   sudo ./scripts/build_image.sh
#
#   perception (Python 3.10)  torch and Depth Anything V2 for the perception node
#   habitat    (Python 3.9)   habitat-sim, which has no 3.10 build
#
# The CUDA runtime comes with the torch wheels and habitat-sim; run with
# `--gpus all` and the NVIDIA Container Toolkit. Scene data is not included;
# the run scripts mount it under
# share/cognav_platform_adapter/simulation/habitat_data.
FROM osrf/ros:humble-desktop

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DEFAULT_TIMEOUT=120
ARG PIP_RETRIES=5
ARG DOWNLOAD_DA2_WEIGHTS=1
# The metric checkpoint; the relative one saturates the metric model's head.
ARG DA2_WEIGHTS_URL="https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"
ARG DA2_WEIGHTS_FILE="depth_anything_v2_metric_hypersim_vits.pth"
ARG DA2_REPO_ARCHIVE_URL="https://codeload.github.com/DepthAnything/Depth-Anything-V2/tar.gz"
ARG DA2_COMMIT="a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        git \
        git-lfs \
        python3-zmq \
        python3-colcon-common-extensions \
        ros-humble-diagnostic-updater \
        ros-humble-irobot-create-description \
        ros-humble-rosidl-default-generators \
        ros-humble-turtlebot4-description \
        ros-humble-irobot-create-msgs \
        ros-humble-nav2-amcl \
        ros-humble-nav2-map-server \
        ros-humble-nav2-lifecycle-manager \
        python3-scipy \
        libgl1 libglvnd0 libegl1 libglib2.0-0 \
        libx11-6 libx11-xcb1 libxau6 libxcb1 libxcb-cursor0 libxcb-icccm4 \
        libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render0 \
        libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-sync1 \
        libxcb-xinerama0 libxcb-xfixes0 libxext6 libxfixes3 libxi6 \
        libxinerama1 libxkbcommon-x11-0 libxrandr2 libxcursor1 \
        libjpeg-dev libpng-dev libsm6 libxrender1 libgomp1 \
        xauth \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV QT_X11_NO_MITSHM=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda install -n base -y conda-libmamba-solver \
    && conda config --set solver libmamba \
    && conda config --set channel_priority strict \
    && conda config --set remote_connect_timeout_secs 30 \
    && conda config --set remote_read_timeout_secs 120 \
    && conda config --set remote_max_retries 5

# perception environment
WORKDIR /tmp
COPY conda/environment.yml .
RUN for attempt in 1 2 3; do \
        conda env remove -n perception -y >/dev/null 2>&1 || true; \
        conda env create -f environment.yml && break; \
        status=$?; \
        if [ "${attempt}" = "3" ]; then exit "${status}"; fi; \
        conda clean -afy; \
        sleep $((attempt * 10)); \
    done \
    && conda run -n perception pip install --no-cache-dir pyzmq h5py \
    && conda clean -afy

# Depth Anything V2 metric-depth sources at a pinned commit.
ENV COGNAV_DA2_SRC=/opt/cognav/vendor
RUN mkdir -p ${COGNAV_DA2_SRC} /tmp/da2 \
    && wget --no-verbose -O /tmp/da2.tar.gz "${DA2_REPO_ARCHIVE_URL}/${DA2_COMMIT}" \
    && tar -xzf /tmp/da2.tar.gz -C /tmp/da2 --strip-components=1 \
    && cp -R /tmp/da2/metric_depth/depth_anything_v2 "${COGNAV_DA2_SRC}/depth_anything_v2" \
    && rm -rf /tmp/da2 /tmp/da2.tar.gz

# habitat environment
RUN conda create -n habitat -y python=3.9 \
    && conda install -n habitat -y -c conda-forge -c aihabitat habitat-sim withbullet \
    && conda install -n habitat -y -c conda-forge git-lfs \
    && conda run -n habitat pip install --no-cache-dir pyzmq \
    && conda clean -afy

# Workspace. The DA2 checkpoint is downloaded unless the checkout has it. PATH
# excludes conda so colcon uses the system Python for message generation.
ENV COGNAV_WS=/opt/ws
WORKDIR ${COGNAV_WS}
COPY . ${COGNAV_WS}/

RUN mkdir -p ${COGNAV_WS}/src/cognav_perception/model_weights \
    && if [ ! -f "${COGNAV_WS}/src/cognav_perception/model_weights/${DA2_WEIGHTS_FILE}" ]; then \
        if [ "${DOWNLOAD_DA2_WEIGHTS}" = "1" ]; then \
            wget --no-verbose -O "${COGNAV_WS}/src/cognav_perception/model_weights/${DA2_WEIGHTS_FILE}" "${DA2_WEIGHTS_URL}"; \
        else \
            echo "Missing DA2 weights: ${COGNAV_WS}/src/cognav_perception/model_weights/${DA2_WEIGHTS_FILE}" >&2; \
            exit 1; \
        fi; \
    fi \
    && . /opt/ros/humble/setup.sh \
    && env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
    && rm -rf ${COGNAV_WS}/build ${COGNAV_WS}/log \
    && rm -rf ${COGNAV_WS}/conda ${COGNAV_WS}/docker

COPY docker/setup_env.sh /opt/cognav/setup_env.sh
COPY docker/entrypoint.sh /opt/entrypoint.sh
COPY docker/run_rviz.sh /opt/cognav/run_rviz.sh
COPY docker/run_rqt_graph.sh /opt/cognav/run_rqt_graph.sh
RUN chmod +x /opt/cognav/setup_env.sh /opt/entrypoint.sh \
        /opt/cognav/run_rviz.sh /opt/cognav/run_rqt_graph.sh \
    && printf '\n. /opt/cognav/setup_env.sh\n' >> /root/.bashrc \
    && printf '. /opt/cognav/setup_env.sh\n' > /etc/profile.d/cognav.sh

# Fail the build if the launch file or a mission did not install.
RUN . ${COGNAV_WS}/install/setup.sh \
    && test -f ${COGNAV_WS}/install/cognav_bt_runner/share/cognav_bt_runner/launch/cognav_bt_launch.py \
    && test -f ${COGNAV_WS}/install/cognav_mission/share/cognav_mission/mission/random_explore.xml \
    && test -f ${COGNAV_WS}/install/cognav_mission/share/cognav_mission/mission/navigate_only.xml \
    && test -f ${COGNAV_WS}/install/cognav_mission/share/cognav_mission/mission/no_retry.xml \
    && test -f ${COGNAV_WS}/install/cognav_mission/share/cognav_mission/mission/no_escape.xml \
    && test -x ${COGNAV_WS}/install/cognav_mission/lib/cognav_mission/mission_node.py

ENV PYTHONPATH=${COGNAV_DA2_SRC}

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["bash"]
