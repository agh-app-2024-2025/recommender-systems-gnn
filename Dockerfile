# Start with a base image that includes Python
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the files needed for installation first (to optimize caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Default command
CMD ["python", "run_cf.py"]
