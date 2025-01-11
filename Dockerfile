# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy dependency files first (to leverage Docker caching)
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry

# Install dependencies (disable project installation)
RUN poetry config virtualenvs.create false && poetry install --only main --no-root

# Copy the rest of the application
COPY . .

# Expose the Flask app's default port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app/flask_app.py"]
