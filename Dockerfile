FROM python:3.11-slim

RUN pip install --upgrade pip

RUN mkdir -p /opt/ml/model
# Set working directory
WORKDIR /opt/program

# Copy 
# Copy inference script and required files
# COPY requirements.txt .
# COPY inference.py .
# COPY model.py .
# COPY HelperFunctions.py .
# COPY random_weights_best_model.pt .
# COPY random_weights_best_state.pt.pt .
# # COPY ModelDefinition2.py .
# copy SageMakerArtifacts/* .
# COPY SageMakerArtifacts/.env .
copy /* .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set up the model directory (SageMaker mounts the model here)

# Expose port 8080 for inference requests
EXPOSE 8080

# Start the Flask application
ENTRYPOINT ["python", "inference.py"]