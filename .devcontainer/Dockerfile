FROM mcr.microsoft.com/devcontainers/python:1-3.9-bullseye

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        curl \
        wget \
        unzip \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3-dev \
        pkg-config \
        && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Node.js for some development tools
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Create workspace directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements*.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -r requirements-dev.txt

# Install additional development tools
RUN pip install \
    jupyterlab \
    pre-commit \
    cookiecutter \
    invoke

# Set up git hooks directory
RUN git config --global init.templateDir /workspace/.git-templates

# Configure shell
RUN echo 'alias ll="ls -la"' >> /home/vscode/.bashrc \
    && echo 'alias la="ls -la"' >> /home/vscode/.bashrc \
    && echo 'alias ..="cd .."' >> /home/vscode/.bashrc \
    && echo 'export PYTHONPATH="/workspace/src:$PYTHONPATH"' >> /home/vscode/.bashrc

# Set ownership
RUN chown -R vscode:vscode /workspace

USER vscode
