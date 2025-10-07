# Use an official lightweight Python image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first
COPY requirements.txt .

# Install dependencies, including gunicorn
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . .

# Expose the port the app will run on
EXPOSE 5000

# This is the Start Command that Render will execute
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
