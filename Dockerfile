FROM python:3.12.1
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e .
EXPOSE 5000
ENV FLASK_APP=application.py
CMD ["python","application.py"]