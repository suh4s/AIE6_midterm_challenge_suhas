FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/home/user/.local/bin:$PATH" \
    HOME=/home/user \
    PYTHONPATH="/home/user/app:$PYTHONPATH"

# Add non-root user
RUN useradd -m -u 1000 user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR $HOME/app

# Copy requirements file
COPY --chown=user requirements.txt .

# Install pip and dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy application files
COPY --chown=user app.py .
COPY --chown=user insight_state.py .
COPY --chown=user chainlit.md .
COPY --chown=user README.md .
COPY --chown=user utils ./utils
COPY --chown=user persona_configs ./persona_configs
COPY --chown=user download_data.py .
COPY --chown=user .env.example .

# Create necessary directories
RUN mkdir -p data data_sources exports public
RUN mkdir -p exports && touch exports/.gitkeep
RUN mkdir -p data && touch data/.gitkeep

# Set permissions
RUN chown -R user:user $HOME

# Switch to non-root user
USER user

# Run data download script to initialize data sources
RUN python download_data.py

# Create config for HF Spaces
RUN mkdir -p $HOME/app/.hf && echo "sdk_version: 3\ntitle: InsightFlow AI\ndescription: Multi-perspective research assistant with visualization capabilities\napp_port: 7860" > $HOME/app/.hf/settings.yaml

# Expose Hugging Face Spaces port
EXPOSE 7860

# Run the app
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]