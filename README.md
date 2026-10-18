# ♻️ EcoScan

**Clasificación de residuos con un modelo preentrenado + recomendación con IA generativa.**
Proyecto grupal · IA Generativa para la Gestión Industrial · UDD

**Integrantes:** Alexis, Gabriela y Agustín

- 🌐 App: https://ecoscan-certamen1-ia-generativa-grupo15.streamlit.app/
- 💻 Repositorio: https://github.com/USUARIO/certamen1-ia-generativa-ecoscan  ← reemplazar USUARIO

## Problema

Separar mal los residuos contamina las cargas de reciclaje. EcoScan permite subir la foto de un residuo, ver qué material es probablemente (cartón, vidrio, metal, papel, plástico u otros residuos) con su nivel de confianza, y recibir una recomendación de Gemini sobre cómo separarlo.

## Datos

| | |
|---|---|
| Dataset | [TrashNet](https://huggingface.co/datasets/garythung/trashnet) (archivo `dataset-resized.zip`) |
| Autores | Gary Thung y Mindy Yang, Stanford CS229 · [repositorio original](https://github.com/garythung/trashnet) |
| Licencia | MIT |
| Clases | cartón 403 · vidrio 501 · metal 410 · papel 594 · plástico 482 · otros residuos 137 (2.527 fotos según la fuente) |
| División | 70 % entrenamiento / 15 % validación / 15 % prueba, estratificada, semilla 42 |
| Limpieza | Se eliminan fotos repetidas y fotos idénticas con etiquetas contradictorias |

**Desbalance:** "otros residuos" tiene ~4 veces menos fotos que papel → usamos accuracy balanceada y pérdida ponderada.
**Limitaciones:** una sola fuente, fondo blanco, un objeto por foto y buena luz; con fotos reales el desempeño puede bajar.

## Modelo

Notebook: [`notebooks/entrenamiento.ipynb`](notebooks/entrenamiento.ipynb) (ejecutar en Google Colab con GPU).

Comparamos tres enfoques con la misma partición y la misma métrica (accuracy balanceada):
1. **CLIP sin entrenar** (línea base zero-shot).
2. **EfficientNet-B0** con transfer learning (timm).
3. **ResNet18** con transfer learning (timm).

El modelo se elige por validación y se reporta en prueba (379 fotos que el modelo nunca vio):

| Enfoque | Accuracy balanceada en prueba |
|---|---:|
| CLIP sin entrenar | 63,9 % |
| ResNet18 | 79,2 % |
| **EfficientNet-B0 (modelo elegido)** | **87,8 %** |

**Modelo elegido: EfficientNet-B0**, por tener la mejor accuracy balanceada en validación; además es liviano (~16 MB). En prueba obtuvo 87,9 % de accuracy. Detalle en `modelo/metricas.json` y en la pestaña *Desempeño* de la app.

**Error más costoso:** que un residuo no reciclable ("otros residuos") se clasifique como reciclable y contamine una carga.

## Estructura

```
app.py                     app web (Streamlit)
notebooks/entrenamiento.ipynb   datos, comparación de modelos, métricas y exportación
modelo/                    modelo.pt, metricas.json y una matriz de confusión por modelo
ejemplos/                  fotos de prueba para la app
.streamlit/config.toml     colores de la app
requirements.txt           librerías
```

## Cómo ejecutar

**Entrenar:** abrir el notebook en Colab → GPU T4 → *Ejecutar todo*. Al final se descarga `para_la_app.zip`, con las carpetas `modelo/` y `ejemplos/`: reemplazarlas en el repositorio.

**Publicar en Streamlit Community Cloud:** *Create app* → este repositorio, rama `main`, archivo `app.py` → *Advanced settings*: **Python 3.11** y el secreto `GEMINI_API_KEY`. Cada commit actualiza la app sola.

**App local** (Python 3.11):
```
pip install -r requirements.txt
streamlit run app.py
```

**Clave de Gemini:** se guarda como secreto `GEMINI_API_KEY` en Streamlit Cloud (*Settings → Secrets*) o en `.streamlit/secrets.toml` en local. **Nunca se sube a GitHub.** A Gemini solo se envía texto (material, confianza y la pregunta del usuario), no la imagen.

## Uso de IA

Usamos ChatGPT/Codex y Claude para generar y simplificar el código del notebook y de la app, y para revisar la documentación. Gemini genera las recomendaciones dentro de la app. El grupo adaptó el código a TrashNet, ejecutó el entrenamiento y revisó los resultados.

⚠️ Prototipo educativo: EcoScan sugiere el material, no certifica que un residuo sea reciclable.
