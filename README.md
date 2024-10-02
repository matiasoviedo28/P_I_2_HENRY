# Proyecto: Análisis de Acceso a Internet en Argentina

## Descripción General

Este proyecto tiene como objetivo realizar un análisis completo del acceso a internet en Argentina. Para esto, me he puesto en el rol de Data Analyst y, a través de un proceso de ETL, EDA y la creación de un dashboard interactivo, pude examinar la evolución de las distintas tecnologías de acceso a internet en el país. 

El análisis incluye la evolución del ancho de banda, el impacto de tecnologías como fibra óptica y la comparación con otros tipos de accesos, usando ejemplos relevantes como el caso de la provincia de San Luis y sus políticas públicas. También se aplicaron proyecciones a futuro para mostrar tendencias claras sobre cómo el mercado está cambiando y hacia dónde se dirige.

## Contenido

- **ETL.ipynb:** Aquí se realiza el proceso de extracción, transformación y carga de los datos. Se identificaron y eliminaron datos redundantes, atípicos y nulos, preparando la información para un análisis más limpio y efectivo.
- **EDA.ipynb:** Se lleva a cabo el análisis exploratorio de datos con el objetivo de identificar tendencias, distribuciones y valores críticos. Las conclusiones de este análisis se utilizaron para diseñar el dashboard.
- **app.py:** Contiene la implementación del dashboard interactivo utilizando Plotly y Dash. Permite a los usuarios explorar los datos, visualizar tendencias y entender la situación del acceso a internet en Argentina.
- **Dashboard:** Se creó un dashboard utilizando 100% codigo python, conecté mis habilidades con redes informaticas, y he realizado un servidor con **Ubuntu Server** Posteriormente he instalado mi repositorio en un entorno virtual y he habilitado el acceso del puerto 8050 a mi IP Pública. [Link al Dashboard](http://201.251.222.112:8050)

## Proceso ETL

La primera etapa fue el proceso ETL (Extract, Transform, Load), donde se trabajó con múltiples datasets que presentaban datos redundantes, nulos y algunas cadenas de texto no deseadas. El principal problema fue organizar estos datasets y limpiar los datos para dejarlos listos para el análisis. 

Se completaron datos faltantes para 2024 y 2025 aplicando un crecimiento estimado del 2% por trimestre, de manera que las proyecciones a futuro fueran consistentes con las tendencias observadas en los datos.

## Análisis Exploratorio de Datos (EDA)

En el EDA se analizó la evolución de las distintas tecnologías de acceso a internet, así como su penetración en las provincias de Argentina. Los datos arrojaron tendencias claras que nos permitieron definir el enfoque del análisis. Algunos puntos clave:

1. **Evolución de la Tecnología:** Se observan tendencias muy definidas para las tecnologías de acceso a internet. Algunas tecnologías, como ADSL, están en clara caída, mientras que la fibra óptica muestra un crecimiento constante.

2. **Análisis Provincial:** Tomamos como ejemplo la provincia de San Luis, que incrementó su media de velocidad de bajada gracias a políticas públicas como "San Luis 1000". Estos análisis locales ayudaron a ilustrar cómo ciertas decisiones pueden impactar en el acceso a internet de una región.

3. **Uso de Plotly:** Decidí utilizar Plotly debido a su capacidad de generar gráficos interactivos que permiten una mejor exploración y análisis de datos, abriendo posibilidades de personalización e integración con herramientas de visualización.

## Dashboard

El dashboard fue desarrollado usando Dash y Plotly. El objetivo era mostrar la información mínima y necesaria para el análisis planteado, ya que la meta era construir un MVP (Producto Viable Mínimo) centrado en la evolución de la conectividad en Argentina. Las secciones principales incluyen:

- **Velocidad Media por Provincia:** Un mapa interactivo que permite visualizar la velocidad media de bajada por provincia.
- **Tendencia de Tecnologías de Acceso:** Incluye un análisis de la fibra óptica por región y la evolución de otras tecnologías.
- **Tendencia de Velocidades de Internet:** Presenta la evolución de distintas velocidades de conexión en cada provincia.
- **Penetración de Accesos por Hogar:** Muestra el crecimiento del acceso a internet en los hogares, permitiendo analizar la adopción de distintas tecnologías.
- **Comparación Banda Ancha Fija vs. Portabilidad:** Proyecciones hasta 2026 que comparan la adopción de banda ancha fija con portabilidad móvil.

## Conclusiones

- **Futuro de las Tecnologías:** Los resultados predicen una caída continua del ADSL debido a sus limitaciones de velocidad, mientras que se espera un crecimiento sostenido de la fibra óptica, impulsado por su capacidad y su costo relativamente bajo.
- **Ejemplo de San Luis:** Analizamos cómo políticas públicas enfocadas pueden incrementar rápidamente la calidad de la conectividad, sirviendo como modelo para otras regiones.

¡Espero que este análisis y las conclusiones presentadas puedan ser útiles para las empresas de telecomunicaciones y para futuras decisiones sobre conectividad en Argentina!
