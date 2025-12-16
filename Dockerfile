# 1. Base Image: Use a slim Python image for a smaller footprint
# We start with a base operating system that includes Python 3.11
FROM python:3.11-slim

# 2. Set the Working Directory
# This is the directory inside the container where your app will live
WORKDIR /app

# 3. Copy Requirements and Install Dependencies
# We need build tools to compile libraries like numpy and FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from your local machine to the container
COPY requirements.txt .

# Install all Python dependencies from the file
RUN pip install --no-cache-dir -r requirements.txt

# 4. Create Directory for Deliverables
# Ensure the required outputs folder exists before we run the code
RUN mkdir -p outputs

# 5. Copy the rest of the application code
# This copies everything else (.py files, agent folders, data folders, etc.)
COPY . .

# 6. Command to Run the System
# This defines the command that executes when the container is run
# It will run your main test script, generating the logs.
CMD ["python", "main.py"]