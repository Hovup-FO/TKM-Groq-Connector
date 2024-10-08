# Usa una imagen base de Python
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo de requisitos (si tienes un requirements.txt)
COPY requirements.txt .

# Instala las dependencias desde el archivo requirements.txt si existe
# O si no tienes uno, instalamos Chainlit directamente
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; else pip install chainlit; fi

# Copia todo el código de la app al contenedor
COPY . .

# Expone el puerto 8000 (que es el predeterminado para Chainlit)
EXPOSE 8000

# Comando para ejecutar Chainlit y asegurarse de que esté escuchando en todas las interfaces
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]

