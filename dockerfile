FROM alyshmahell/oltralotus:base

WORKDIR /apps/oltralotus 

COPY oltralotus/ ./

RUN mv ./script.py /usr/local/bin/oltralotus && chmod +x /usr/local/bin/oltralotus

CMD python server.py