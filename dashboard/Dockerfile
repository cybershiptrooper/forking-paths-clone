FROM python:3.12-slim
WORKDIR /app
COPY dashboard/ dashboard/
COPY results/ results/
EXPOSE 10000
CMD ["python", "dashboard/serve.py"]
