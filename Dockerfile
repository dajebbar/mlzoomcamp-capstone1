FROM python:3.9-slim

ENV PYTHONUNBUFFERED=TRUE

RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system && rm -rf /root/.cache

COPY ["predict.py", "kitchenwareModel.h5", "banner.png", "./"]

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]

CMD ["predict.py"]