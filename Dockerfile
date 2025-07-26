# use official python base image
FROM python:3.10-slim

# working dir
WORKDIR /app

# copy files
COPY . /app

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# expose port
EXPOSE 8100

# command to run server
CMD ["uvicorn","iris_api_w6:app","--host","0.0.0.0","--port","8100","--reload"]
