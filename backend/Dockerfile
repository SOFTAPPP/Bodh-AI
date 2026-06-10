FROM python:3.10-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application
COPY ./ /code/

# Create data directory with proper permissions for Hugging Face
RUN mkdir -p /code/data && chmod -R 777 /code/data

# Hugging Face Spaces require running on port 7860
ENV PORT=7860

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
