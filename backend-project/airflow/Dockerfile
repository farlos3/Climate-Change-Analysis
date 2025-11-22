FROM apache/airflow:3.1.0-python3.12

# Copy requirements
COPY requirements.txt /requirements.txt

# Install python packages using airflow user
USER airflow
RUN pip install --no-cache-dir -r /requirements.txt

# กลับมาเป็น root ถ้าต้องติดตั้ง apt
USER root
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# กลับเป็น airflow (สำคัญมาก)
USER airflow