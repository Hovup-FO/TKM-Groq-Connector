# Usa una imagen base de Python
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo de requisitos a la imagen
COPY requirements.txt .

# Instala las dependencias desde el archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el código de la app al contenedor
COPY . .

# Expone el puerto que usará Chainlit (el puerto predeterminado es 8000)
EXPOSE 8000

# Comando para ejecutar Chainlit al iniciar el contenedor
CMD ["chainlit", "run", "app.py"]
