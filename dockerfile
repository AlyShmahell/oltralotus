FROM oltralytics-base

WORKDIR /apps/oltralytics 

COPY oltralytics/ ./

RUN mv ./script.py /usr/local/bin/oltralytics && chmod +x /usr/local/bin/oltralytics

CMD python server.py