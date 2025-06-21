# Start with the official Azure Functions Python 3.9 image
FROM mcr.microsoft.com/azure-functions/python:3.9-python3.9-appservice

# Set environment variable for the port
ENV PORT=8000

# Set the working directory
WORKDIR /home/site/wwwroot

# Copy the application files to the container
COPY . /home/site/wwwroot

# Install dependencies in the virtual environment
RUN python -m venv antenv && \
    . antenv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Define the command to start your app
CMD ["bash", "startup.sh"]
