# Base image
FROM continuumio/miniconda3

# Install python packages
RUN mkdir /opt/api
COPY requirements.txt /opt/api
RUN pip install -r /opt/api/requirements.txt

# Copy files into container
COPY swagger /opt/api/swagger
COPY model /opt/api/model
COPY youtube_api.py /opt/api

# Set work directory and open required port
WORKDIR /opt/api
EXPOSE 8080

# Run service script
CMD ["python","youtube_api.py"]
