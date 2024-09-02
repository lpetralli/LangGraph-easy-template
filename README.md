# LangGraph-easy-template


Este repositorio contiene un chatbot basado en un agente inteligente que utiliza modelos de lenguaje y herramientas de recuperación de información.

## Pasos de instalación y uso

1. **Instalar requisitos:**
   ```
   pip install -r requirements.txt
   ```

2. **Crear archivo .env:**
   Crea un archivo `.env` en la raíz del proyecto con tus propias variables de entorno:
   ```
   GROQ_API_KEY=tu_clave_api_de_groq
   OPENAI_API_KEY=tu_clave_api_de_openai
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=tu_clave_api_de_langsmith
   LANGSMITH_PROJECT="LangGraph-easy-template"
   ```

3. **Crear la vector store local:**
   Utiliza el notebook `test-RAG.ipynb` para generar la base de conocimientos local.

4. **Modificar el prompt (opcional):**
   Si lo deseas, puedes modificar el prompt en el archivo `agent.py`.

5. **Elegir el proveedor del modelo:**
   En `chat.py`, selecciona 'openai' o 'groq' como proveedor del modelo de lenguaje.

6. **Ejecutar el chat:**
   Inicia la aplicación con el siguiente comando:
   ```
   streamlit run chat.py
   ```

Este chatbot utiliza LangChain y LangGraph para estructurar el flujo de conversación y las capacidades del agente, permitiendo interacciones basadas en modelos de lenguaje avanzados y búsquedas en una base de conocimientos local.