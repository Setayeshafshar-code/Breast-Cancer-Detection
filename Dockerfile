# Use an official Python image as the base
FROM python:3.9-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file to the container
COPY requirements.txt requirements.txt

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 5000 for the Flask application
EXPOSE 5000

# Command to run the application
CMD ["python3", "app.py"]
