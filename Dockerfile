FROM ubuntu:latest
LABEL authors="SHPS"

ENTRYPOINT ["top", "-b"]