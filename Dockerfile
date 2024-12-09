# 使用Python 3.10作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 安装系统依赖
RUN echo "deb http://deb.debian.org/debian bookworm main" > /etc/apt/sources.list && \
    sed -i 's|http://deb.debian.org/debian|http://mirrors.aliyun.com/debian|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    curl && \
    rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY app.py .
COPY README.md .

# 创建数据存储目录
RUN mkdir -p saved_models saved_datasets logs

# 安装Python依赖i
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir gunicorn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 暴露端口
EXPOSE 5000

# 启动命令（使用gunicorn）
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "app:app"] 
