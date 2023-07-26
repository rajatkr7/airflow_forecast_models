FROM python:3.8.9

ADD main.py .

COPY requirements.txt ./

RUN pip3 install --no-cache-dir --upgrade pip \
  && pip3 install --no-cache-dir -r requirements.txt \
  && python -m pip install statsmodels

COPY . .

CMD ["python3", "./main.py"]