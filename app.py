"""
Lovable Prompt Generator - Streamlit App

Genera prompts estructurados para Lovable a partir de:
- Capturas del brandboard/manual de marca
- Capturas del copy aprobado

Usa Claude Vision (via OpenRouter) para extraer información.
"""

import streamlit as st
import base64
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno (local)
load_dotenv()

# Obtener API key (primero intenta Streamlit secrets, luego env vars)
def get_api_key():
    # Primero intenta Streamlit secrets
    if "OPENROUTER_API_KEY" in st.secrets:
        return st.secrets["OPENROUTER_API_KEY"]
    # Luego variables de entorno
    return os.getenv("OPENROUTER_API_KEY")

# Cliente se inicializa de forma perezosa
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = get_api_key()
        if not api_key:
            st.error("⚠️ No se encontró OPENROUTER_API_KEY. Configura los Secrets en Streamlit Cloud.")
            st.stop()
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
    return _client

# Modelo de Claude con visión
VISION_MODEL = "anthropic/claude-sonnet-4"

# Plantilla base para el prompt de Lovable (actualizada)
PLANTILLA_LOVABLE = """Crea una landing page de registro para webinar usando las imágenes adjuntas como referencia:
- Imagen del brandboard → para colores, tipografía y estilo visual
- Imágenes de estructura (opcional) → para ver layouts de secciones

DATOS TÉCNICOS:
- Color primario: {color_primario}
- Color secundario: {color_secundario}
- Color de texto: {color_texto}
- Color de fondo: {color_fondo}
- Tipografía: {tipografia}
- Estilo: {estilo}
- Botones: fondo {color_secundario}, texto blanco, bordes redondeados

IMPORTANTE: NO modifiques el texto que te proporciono. Cópialo exactamente palabra por palabra, sin cambiar ni una coma.

SECCIONES:

{secciones}

NOTAS TÉCNICAS:
- Mobile-first, responsive
- Donde indique FOTO, IMAGEN o MOCKUP, dejar placeholder gris
- Formulario con validación básica
- Contadores regresivos funcionales si aplica
- Scroll suave entre secciones

RECORDATORIO: No modifiques el texto. Cada palabra, cada coma, cada punto debe ser exacto."""


def encode_image_to_base64(uploaded_file):
    """Convierte un archivo subido a base64."""
    return base64.standard_b64encode(uploaded_file.read()).decode("utf-8")


def get_image_media_type(filename):
    """Determina el tipo MIME de la imagen."""
    ext = filename.lower().split('.')[-1]
    types = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    return types.get(ext, 'image/png')


def extract_brand_info(brandboard_images):
    """Extrae información de marca del brandboard usando Claude Vision."""

    # Construir mensaje con imágenes
    content = [
        {
            "type": "text",
            "text": """Analiza este brandboard/manual de marca y extrae la siguiente información en formato JSON:

{
    "color_primario": "#HEXCODE (el color principal de la marca)",
    "color_secundario": "#HEXCODE (el color secundario o de acento, suele ser para botones/CTAs)",
    "color_texto": "descripción del color de texto (ej: 'gris azulado oscuro', '#333333')",
    "color_fondo": "descripción del fondo (ej: 'blanco', 'crema claro', '#FAFAFA')",
    "tipografia": "Nombre EXACTO de la tipografía (debe existir en Google Fonts)",
    "estilo": "palabras clave que describan el estilo (ej: 'wellness, profesional, cercano, limpio')",
    "notas_adicionales": "cualquier otra observación relevante sobre el estilo visual"
}

IMPORTANTE sobre tipografía:
- Busca el nombre exacto de la fuente en el brandboard
- Debe ser una fuente que exista en Google Fonts
- Ejemplos válidos: "Be Vietnam Pro", "Montserrat", "Poppins", "Inter", "Playfair Display"

Si no puedes identificar algo con certeza, indica "NO IDENTIFICADO".
Responde SOLO con el JSON, sin explicaciones adicionales."""
        }
    ]

    # Añadir imágenes
    for img in brandboard_images:
        img.seek(0)  # Reset file pointer
        base64_img = encode_image_to_base64(img)
        media_type = get_image_media_type(img.name)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{base64_img}"
            }
        })

    response = get_client().chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=1000
    )

    # Parsear respuesta JSON
    response_text = response.choices[0].message.content.strip()
    # Limpiar si viene con ```json
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
    response_text = response_text.strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {
            "color_primario": "NO IDENTIFICADO",
            "color_secundario": "NO IDENTIFICADO",
            "tipografia": "NO IDENTIFICADO",
            "estilo_botones": "NO IDENTIFICADO",
            "notas_adicionales": response_text
        }


def extract_copy(copy_images):
    """Extrae el copy textual de las capturas usando Claude Vision."""

    content = [
        {
            "type": "text",
            "text": """Transcribe TODO el texto visible en estas imágenes.

REGLAS CRÍTICAS:
1. Transcribe PALABRA POR PALABRA, exactamente como aparece
2. NO resumas, NO parafrasees, NO "mejores" el texto
3. Mantén saltos de línea donde los veas
4. Si hay secciones claras, sepáralas con una línea en blanco
5. Incluye TODO: títulos, subtítulos, bullets, botones, disclaimers

El copy de marketing está aprobado por el cliente y NO puede modificarse "ni una coma".

Responde SOLO con el texto transcrito, sin comentarios tuyos."""
        }
    ]

    # Añadir imágenes
    for img in copy_images:
        img.seek(0)
        base64_img = encode_image_to_base64(img)
        media_type = get_image_media_type(img.name)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{base64_img}"
            }
        })

    response = get_client().chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=4000
    )

    return response.choices[0].message.content.strip()


def structure_copy_into_sections(raw_copy):
    """Organiza el copy extraído en secciones numeradas para Lovable."""

    response = get_client().chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": f"""Organiza el siguiente copy de landing page en secciones numeradas para Lovable.

COPY EXTRAÍDO:
{raw_copy}

FORMATO DE SALIDA - Usa EXACTAMENTE este formato con secciones numeradas:

### SECCIÓN 1: HERO (centrado)
[Título principal]
[Subtítulo]
[Texto descriptivo]
Formulario:
- Campo: Nombre
- Campo: Email
- Checkbox: [texto del checkbox si existe]
- Botón: [texto del botón]

### SECCIÓN 2: URGENCIA (centrado)
[Fecha del evento]
[Contador regresivo]
[Texto de urgencia]

### SECCIÓN 3: LEAD MAGNET (centrado)
[Mockup regalo: descripción]
[Descripción del regalo/bonus]

### SECCIÓN 4: PAIN POINTS (centrado)
Título: [Esto es para ti si... o similar]
- [punto 1]
- [punto 2]
- [etc.]

### SECCIÓN 5: BENEFICIOS + BIO (2 columnas en paralelo)
COLUMNA IZQUIERDA:
Título: [Lo que vas a aprender/descubrir]
- [beneficio 1]
- [beneficio 2]
- [etc.]

COLUMNA DERECHA:
[PLACEHOLDER: FOTO DEL EXPERTO]
Nombre: [nombre]
Título: [credenciales]
Bio: [texto completo de la bio]

### SECCIÓN 6: TESTIMONIOS (centrado o grid de 3 columnas)
[Testimonio 1]
[Testimonio 2]
[etc.]

### SECCIÓN 7: CTA FINAL (centrado)
[Título de cierre]
[Fecha recordatorio]
[Botón: texto del botón]

REGLAS CRÍTICAS:
1. Copia el texto EXACTAMENTE como está - PALABRA POR PALABRA
2. NO resumas, NO parafrasees, NO "mejores" el texto
3. Si una sección no existe en el copy, OMÍTELA (no la incluyas)
4. Indica fotos con [PLACEHOLDER: descripción]
5. NO inventes contenido que no esté en el copy original
6. Para secciones de 2 columnas, especifica claramente qué va en cada columna"""
        }],
        max_tokens=4000
    )

    return response.choices[0].message.content.strip()


def generate_lovable_prompt(brand_info, structured_sections):
    """Genera el prompt final para Lovable."""

    return PLANTILLA_LOVABLE.format(
        color_primario=brand_info.get("color_primario", "#000000"),
        color_secundario=brand_info.get("color_secundario", "#666666"),
        color_texto=brand_info.get("color_texto", "gris oscuro"),
        color_fondo=brand_info.get("color_fondo", "blanco"),
        tipografia=brand_info.get("tipografia", "Inter"),
        estilo=brand_info.get("estilo", "moderno, limpio, profesional"),
        secciones=structured_sections
    )


# ============== INTERFAZ STREAMLIT ==============

st.set_page_config(
    page_title="Generador de Prompts para Lovable",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Generador de Prompts para Lovable")
st.markdown("Sube las capturas del brandboard y del copy para generar un prompt estructurado.")

# Columnas para subir archivos
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Brandboard / Manual de Marca")
    st.caption("Sube capturas del manual de marca (colores, tipografía, estilo)")
    brandboard_files = st.file_uploader(
        "Arrastra o selecciona imágenes",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="brandboard"
    )
    if brandboard_files:
        st.success(f"✅ {len(brandboard_files)} imagen(es) subida(s)")

with col2:
    st.subheader("📝 Copy de la Landing")
    st.caption("Sube capturas del copy aprobado (todas las secciones)")
    copy_files = st.file_uploader(
        "Arrastra o selecciona imágenes",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="copy"
    )
    if copy_files:
        st.success(f"✅ {len(copy_files)} imagen(es) subida(s)")

st.divider()

# Botón para generar
if st.button("🚀 Generar Prompt para Lovable", type="primary", use_container_width=True):

    if not brandboard_files:
        st.error("⚠️ Sube al menos una imagen del brandboard")
    elif not copy_files:
        st.error("⚠️ Sube al menos una imagen del copy")
    else:
        with st.spinner("Procesando imágenes con Claude Vision..."):

            # Paso 1: Extraer info del brandboard
            st.info("🔍 Analizando brandboard...")
            try:
                brand_info = extract_brand_info(brandboard_files)

                # Mostrar info extraída
                with st.expander("📊 Información de marca extraída", expanded=True):
                    st.json(brand_info)

            except Exception as e:
                st.error(f"Error al analizar brandboard: {str(e)}")
                st.stop()

            # Paso 2: Extraer copy
            st.info("📝 Transcribiendo copy...")
            try:
                raw_copy = extract_copy(copy_files)

                with st.expander("📄 Copy extraído (texto crudo)", expanded=False):
                    st.text(raw_copy)

            except Exception as e:
                st.error(f"Error al extraer copy: {str(e)}")
                st.stop()

            # Paso 3: Estructurar en secciones
            st.info("🏗️ Estructurando secciones...")
            try:
                structured_sections = structure_copy_into_sections(raw_copy)
            except Exception as e:
                st.error(f"Error al estructurar secciones: {str(e)}")
                st.stop()

            # Paso 4: Generar prompt final
            st.info("✨ Generando prompt final...")
            final_prompt = generate_lovable_prompt(brand_info, structured_sections)

        # Mostrar resultado
        st.success("✅ ¡Prompt generado!")

        st.subheader("📋 Prompt para Lovable")
        st.text_area(
            "Copia este prompt y pégalo en Lovable junto con las imágenes del brandboard",
            value=final_prompt,
            height=500,
            key="result"
        )

        # Botón para copiar (usando JavaScript)
        st.markdown("""
        <script>
        function copyToClipboard() {
            const text = document.querySelector('textarea').value;
            navigator.clipboard.writeText(text);
        }
        </script>
        """, unsafe_allow_html=True)

        st.download_button(
            label="📥 Descargar prompt como archivo",
            data=final_prompt,
            file_name="prompt_lovable.txt",
            mime="text/plain"
        )

# Footer
st.divider()
st.caption("💡 **Tip:** Recuerda subir también las imágenes del brandboard a Lovable junto con este prompt.")
