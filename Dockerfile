# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for Google Cloud Run
EXPOSE 8080

# Set environment variable to tell Flask it's in production
ENV FLASK_ENV=production

# Set the PORT environment variable for Cloud Run
ENV PORT=8080

# Run gunicorn to serve the Flask app, binding to the PORT environment variable
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]

