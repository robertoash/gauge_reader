FROM python:3
ADD gauge_reader.py /
ADD secrets.py /
ADD gauge.jpg /
RUN pip install paho-mqtt
RUN pip install numpy==1.21.5
RUN pip install pandas
CMD [ "python", "./gauge_reader.py" ]
