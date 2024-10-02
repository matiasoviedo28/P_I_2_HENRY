import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
import plotly.graph_objs as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
import os

ano_actual = datetime.now().year

#inicializar la app dash con un tema de bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Dashboard'

#leer los datos desde los archivos csv y geojson
carpeta = "datasets/"
df_velocidad = pd.read_csv(carpeta + 'Internet_Velocidad_%_por_prov.csv')
gdf = gpd.read_file(carpeta + 'mapas/map.geojson')
df_tecnologia = pd.read_csv(carpeta + 'Internet_Accesos_Por_Tecnologia.csv')
df_vel_tecnologia = pd.read_csv(carpeta + 'Internet_Accesos_Por_velocidad.csv')
df_accesos_hogares = pd.read_csv(carpeta + 'Internet_Penetracion-hogares.csv')
df_portabilidad = pd.read_csv(carpeta + 'Portabilidad_Portin.csv')
df_portabilidad.drop(columns=['nextel'], inplace=True)

#completar datos faltantes para 2024 y 2025 con un crecimiento del 2% por trimestre
def completar_datos(df, cols):
    df_2024 = df[df['año'] == 2024].groupby(['provincia']).mean(numeric_only=True).reset_index()
    nuevos_datos = []
    for provincia in df_2024['provincia'].unique():
        row_2024_q1 = df_2024[df_2024['provincia'] == provincia].iloc[0]
        for year in [2024, 2025]:
            trimestres = [2, 3, 4] if year == 2024 else [1, 2, 3, 4]
            for trimestre in trimestres:
                row_2024_q1[cols] *= 1.02
                nuevos_datos.append({
                    'año': year,
                    'trimestre': trimestre,
                    'provincia': provincia,
                    **{col: row_2024_q1[col] for col in cols},
                    'total': row_2024_q1[cols].sum()
                })
    df_nuevos = pd.DataFrame(nuevos_datos)
    df_completo = pd.concat([df, df_nuevos], ignore_index=True)
    return df_completo

#completar los datos del dataframe original
df_tecnologia = completar_datos(df_tecnologia, ['adsl', 'cablemodem', 'fibra_óptica', 'wireless', 'otros'])
df_vel_tecnologia = completar_datos(df_vel_tecnologia, ['hasta_512_kbps', '+_512_kbps_-_1_mbps', '+_1_mbps_-_6_mbps',
                                                        '+_6_mbps_-_10_mbps', '+_10_mbps_-_20_mbps', '+_20_mbps_-_30_mbps',
                                                        '+_30_mbps', 'otros'])

#promediar por ano para los graficos de lineas
df_tecnologia_yearly = df_tecnologia.groupby(['año', 'provincia']).mean(numeric_only=True).reset_index()
df_vel_tecnologia_yearly = df_vel_tecnologia.groupby(['año', 'provincia']).mean(numeric_only=True).reset_index()
df_accesos_hogares_yearly = df_accesos_hogares.groupby(['provincia', 'año']).mean(numeric_only=True).reset_index()

df_tecnologia = pd.read_csv('datasets/Internet_Accesos_Por_Tecnologia.csv')

#crear un diccionario que asocie las provincias con sus regiones
provincias_regiones = {
    'Buenos Aires': 'Pampeana', 'Capital Federal': 'Pampeana', 'Catamarca': 'Noroeste',
    'Chaco': 'Noreste', 'Chubut': 'Patagonia', 'Córdoba': 'Pampeana', 'Corrientes': 'Noreste',
    'Entre Ríos': 'Pampeana', 'Formosa': 'Noreste', 'Jujuy': 'Noroeste', 'La Pampa': 'Patagonia',
    'La Rioja': 'Noroeste', 'Mendoza': 'Cuyo', 'Misiones': 'Noreste', 'Neuquén': 'Patagonia',
    'Río Negro': 'Patagonia', 'Salta': 'Noroeste', 'San Juan': 'Cuyo', 'San Luis': 'Cuyo',
    'Santa Cruz': 'Patagonia', 'Santa Fe': 'Pampeana', 'Santiago del Estero': 'Noroeste',
    'Tierra del Fuego': 'Patagonia', 'Tucumán': 'Noroeste'
}

#configuracion general del diseno de la app
app.layout = dbc.Container([
    html.H1('Dashboard de Internet en Argentina', style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Velocidad Media por Provincia', children=[
            html.Div([
                html.Label('Seleccionar Año:'),
                dcc.Dropdown(
                    id='dropdown-year-speed',
                    options=[{'label': str(year), 'value': year} for year in df_velocidad['año'].unique()],
                    value=df_velocidad['año'].max(),
                    multi=False,
                    clearable=False,
                    style={'width': '50%'}
                ),
                dcc.Graph(id='choropleth-map', style={'height': '570px', 'width': '100%'})
            ], style={'backgroundColor': '#f2f2f2', 'padding': '20px'})
        ]),
        dcc.Tab(label='Tendencia de Tecnologías de Acceso', children=[
            html.Div([
                html.Label('Seleccionar Región:', style={'fontSize': '75%'}),
                dcc.Dropdown(
                    id='dropdown-region-fibra',
                    options=[{'label': 'Todos', 'value': 'Todos'}] + [{'label': region, 'value': region} for region in provincias_regiones.values()],
                    value='Todos',
                    multi=False
                ),
                html.Div([
                    dcc.Graph(id='scatter-tech', style={'flex': '1'}),
                    dcc.Graph(id='line-fibra-optica', style={'flex': '1'})                    
                ], style={'display': 'flex', 'justify-content': 'space-between'}),
            ])
        ]),
        dcc.Tab(label='Tendencia de Velocidades de Internet', children=[
            html.Div([
                html.Label('Seleccionar Provincia:'),
                dcc.Dropdown(
                    id='dropdown-province-speed-tech',
                    options=[{'label': 'Todos', 'value': 'Todos'}] + [{'label': prov, 'value': prov} for prov in df_vel_tecnologia['provincia'].unique()],
                    value='Todos',
                    multi=False
                ),
                dcc.Graph(id='scatter-speed-tech')
            ])
        ]),
        dcc.Tab(label='Penetración de Accesos por Hogar', children=[
            html.Div([
                html.Label('Seleccionar Provincia:'),
                dcc.Dropdown(
                    id='dropdown-province-household',
                    options=[{'label': 'Todos', 'value': 'Todos'}] + [{'label': prov, 'value': prov} for prov in df_accesos_hogares['provincia'].unique()],
                    value='Todos',
                    multi=False
                ),
                dcc.Graph(id='line-household-access')
            ])
        ]),
        dcc.Tab(label='Evolución de la Velocidad Media', children=[
            html.Div([
                html.Label('Seleccionar Provincia:', style={'fontSize': '75%'}),
                dcc.Dropdown(
                    id='dropdown-province-mbps',
                    options=[{'label': 'Todos', 'value': 'Todos'}] + [{'label': prov, 'value': prov} for prov in df_velocidad['provincia'].unique()],
                    value='Todos',
                    multi=False
                ),
                dcc.Graph(id='line-mbps')
            ])
        ]),
        dcc.Tab(label='Comparación Banda Ancha Fija vs Portabilidad', children=[
            html.Div([
                dcc.Graph(id='comparison-graph', style={'display': 'inline-block', 'width': '49%'}),
                dcc.Graph(id='pie-chart-portabilidad', style={'display': 'inline-block', 'width': '49%'})
            ])
        ])
    ])
], fluid=True)


#establecer suppress_callback_exceptions para evitar errores si hay elementos generados dinamicamente
app.config.suppress_callback_exceptions = True


#callback para actualizar el grafico de velocidad de internet por provincia
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('dropdown-year-speed', 'value')]
)
def update_map(selected_year):
    #filtrar datos del ano seleccionado
    df_filtered = df_velocidad[df_velocidad['año'] == selected_year]

    #agrupar por provincia y calcular el promedio de bajada
    df_yearly = df_filtered.groupby('provincia')['mbps_(media_de_bajada)'].mean().reset_index()

    #reorganizar valores nulos y asignar velocidades de bajada al geodataframe
    gdf['mbps_(media_de_bajada)'] = gdf['nombre'].map(df_yearly.set_index('provincia')['mbps_(media_de_bajada)'])
    gdf['mbps_(media_de_bajada)'] = gdf['mbps_(media_de_bajada)'].fillna(0)

    #crear el grafico
    fig = go.Figure()

    #anadir geojson como un mapa coropletico
    fig.add_trace(go.Choroplethmapbox(
        geojson=gdf.__geo_interface__,
        locations=gdf.index,
        z=gdf['mbps_(media_de_bajada)'],
        colorscale='Viridis',
        zmin=0,
        zmax=gdf['mbps_(media_de_bajada)'].max(),
        marker_opacity=0.7,
        marker_line_width=0.5,
        marker_line_color='black',
        colorbar=dict(title='Mbps (media de bajada)'),
        hoverinfo='text',
        text=gdf['nombre'] + ': ' + gdf['mbps_(media_de_bajada)'].astype(str) + ' Mbps',
    ))

    #configurar la posicion del mapa
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=2.8,
        mapbox_center={"lat": -38.4161, "lon": -63.6167},  #centro de argentina
        title_text=f'Velocidad Media de Internet por Provincia en Argentina ({selected_year})',
    )

    return fig

#callback para actualizar el grafico de tecnologias de acceso
@app.callback(
    Output('scatter-tech', 'figure'),
    [Input('dropdown-region-fibra', 'value')]
)
def update_scatter_tech(selected_province):
    #filtrar datos por provincia seleccionada
    #print(f"provincia seleccionada: {selected_province}")
    
    if selected_province == 'Todos':
        df_filtered = df_tecnologia_yearly.groupby('año').sum(numeric_only=True).reset_index()
    else:
        df_filtered = df_tecnologia_yearly[df_tecnologia_yearly['provincia'] == selected_province]
    
    #print(f"datos filtrados:\n{df_filtered.head()}")

    #crear grafico de lineas para las tecnologias
    fig = px.line(df_filtered, x='año', y=['adsl', 'cablemodem', 'fibra_óptica', 'wireless', 'otros'],
                  labels={'value': 'Accesos', 'año': 'Año'},
                  title=f'Tendencia de Accesos por Tecnología en {"todas las provincias" if selected_province == "Todos" else selected_province}',
                  line_shape='spline')

    #anadir la linea negra para el total
    fig.add_trace(go.Scatter(
        x=df_filtered['año'],
        y=df_filtered[['adsl', 'cablemodem', 'fibra_óptica', 'wireless', 'otros']].sum(axis=1),
        mode='lines+markers',
        name='Total',
        line=dict(color='black', width=2)
    ))

    #print(f"grafico generado: {fig}")

    return fig


#callback para actualizar el grafico de velocidades de acceso
@app.callback(
    Output('scatter-speed-tech', 'figure'),
    [Input('dropdown-province-speed-tech', 'value')]
)
def update_scatter_speed_tech(selected_province):
    #filtrar datos por provincia seleccionada
    if selected_province == 'Todos':
        df_filtered = df_vel_tecnologia_yearly.groupby('año').sum(numeric_only=True).reset_index()
    else:
        df_filtered = df_vel_tecnologia_yearly[df_vel_tecnologia_yearly['provincia'] == selected_province]

    #crear grafico de lineas para las velocidades
    fig = px.line(df_filtered, x='año', y=['hasta_512_kbps', '+_512_kbps_-_1_mbps', '+_1_mbps_-_6_mbps',
                                           '+_6_mbps_-_10_mbps', '+_10_mbps_-_20_mbps', '+_20_mbps_-_30_mbps',
                                           '+_30_mbps', 'otros'],
                  labels={'value': 'Accesos', 'año': 'Año'},
                  title=f'Tendencia de Velocidades de Internet en {"todas las provincias" if selected_province == "Todos" else selected_province}',
                  line_shape='spline')

    #anadir la linea negra para el total
    fig.add_trace(go.Scatter(
        x=df_filtered['año'],
        y=df_filtered[['hasta_512_kbps', '+_512_kbps_-_1_mbps', '+_1_mbps_-_6_mbps',
                       '+_6_mbps_-_10_mbps', '+_10_mbps_-_20_mbps', '+_20_mbps_-_30_mbps',
                       '+_30_mbps', 'otros']].sum(axis=1),
        mode='lines+markers',
        name='Total',
        line=dict(color='black', width=2)
    ))

    return fig

#callback para actualizar el grafico de penetracion de accesos por hogar
@app.callback(
    Output('line-household-access', 'figure'),
    [Input('dropdown-province-household', 'value')]
)
def update_line_household_access(selected_province):
    #filtrar datos por provincia seleccionada
    if selected_province == 'Todos':
        df_filtered = df_accesos_hogares_yearly.groupby('año').sum(numeric_only=True).reset_index()
    else:
        df_filtered = df_accesos_hogares_yearly[df_accesos_hogares_yearly['provincia'] == selected_province]

    #calcular el kpi de aumento del 2% por trimestre para el ultimo valor registrado (primer trimestre de 2024)
    if not df_filtered.empty and df_filtered['año'].max() == 2024:
        last_value = df_filtered[df_filtered['año'] == 2024]['accesos_por_cada_100_hogares'].values[-1]
        kpi_values = [last_value * (1.02 ** i) for i in range(1, 5)]  #trimestres 2, 3, 4 de 2024 y 1 de 2025

        #crear dataframe para los kpi
        kpi_df = pd.DataFrame({
            'año': [2024, 2024, 2024, 2025],
            'trimestre': [2, 3, 4, 1],
            'accesos_por_cada_100_hogares': kpi_values
        })

        #anadir los kpi al dataframe filtrado
        df_filtered = pd.concat([df_filtered, kpi_df])

    #crear el grafico de lineas
    fig = px.line(
        df_filtered,
        x='año',
        y='accesos_por_cada_100_hogares',
        labels={'accesos_por_cada_100_hogares': 'Accesos por cada 100 Hogares', 'año': 'Año'},
        title=f'Penetración de Internet en {"todas las provincias" if selected_province == "Todos" else selected_province}',
        line_shape='spline'
    )

    #anadir una linea para los objetivos kpi
    fig.add_trace(go.Scatter(
        x=kpi_df['año'],
        y=kpi_df['accesos_por_cada_100_hogares'],
        mode='lines+markers',
        name='Objetivo KPI 2%',
        line=dict(color='green', width=2, dash='dash')
    ))

    return fig

#callback para el grafico de evolucion de la velocidad media
@app.callback(
    Output('line-mbps', 'figure'),
    [Input('dropdown-province-mbps', 'value')]
)
def update_line_mbps(selected_province):
    #filtrar datos por provincia seleccionada
    if selected_province == 'Todos':
        df_filtered = df_velocidad.groupby('año').agg({'mbps_(media_de_bajada)': 'mean'}).reset_index()
    else:
        df_filtered = df_velocidad[df_velocidad['provincia'] == selected_province].groupby('año').agg({'mbps_(media_de_bajada)': 'mean'}).reset_index()

    #transformar el objetivo para un ajuste exponencial
    df_filtered['log_mbps'] = np.log(df_filtered['mbps_(media_de_bajada)'])
    X = df_filtered['año'].values.reshape(-1, 1)
    y_log = df_filtered['log_mbps'].values
    model = LinearRegression()
    model.fit(X, y_log)

    #predecir el valor logaritmico para 2024 y convertirlo de nuevo a su escala original
    predicted_log_mbps_2024 = model.predict([[ano_actual]])
    predicted_mbps_2024 = np.exp(predicted_log_mbps_2024)

    #eliminar cualquier punto existente para el ano actual (2024)
    df_filtered = df_filtered[df_filtered['año'] != ano_actual]

    #agregar la prediccion de 2024 al dataframe
    df_predicted_2024 = pd.DataFrame({'año': [ano_actual], 'mbps_(media_de_bajada)': predicted_mbps_2024})
    df_combined = pd.concat([df_filtered[['año', 'mbps_(media_de_bajada)']], df_predicted_2024])

    #crear el grafico de lineas
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_combined['año'], 
        y=df_combined['mbps_(media_de_bajada)'], 
        mode='lines+markers', 
        name='Media de Mbps',
        line=dict(color='blue'),
        marker=dict(size=8)
    ))

    #personalizar el grafico
    fig.update_layout(
        title=f'Evolución de la Velocidad Media de Internet en {"todas las provincias" if selected_province == "Todos" else selected_province}',
        xaxis_title='Año',
        yaxis_title='Mbps (Media de Bajada)',
        hovermode='x',
        template='plotly_white'
    )
    return fig

#callback para el grafico de comparacion (banda ancha vs portabilidad)
@app.callback(
    Output('comparison-graph', 'figure'),
    Input('comparison-graph', 'id')  #dummy input, solo para actualizar el grafico
)
def update_comparison_graph(_):
    #filtrar y preparar datos para el grafico
    df_tecnologia_filtered = df_tecnologia[df_tecnologia['año'] < 2024]
    df_portabilidad_filtered = df_portabilidad[df_portabilidad['año'] < 2024].copy()  #usar .copy() para evitar el settingwithcopywarning

    #agrupar y sumar por ano para banda ancha fija y portabilidad
    df_fija = df_tecnologia_filtered.groupby('año')['total'].sum().reset_index()
    df_fija = df_fija.rename(columns={'total': 'total_fija'})
    
    #calcular el total de portabilidad
    df_portabilidad_filtered['total_portabilidad'] = df_portabilidad_filtered[['personal', 'claro', 'movistar']].sum(axis=1)
    df_portabilidad_filtered = df_portabilidad_filtered.groupby('año')['total_portabilidad'].sum().reset_index()

    #unir ambos dataframes y calcular proyecciones
    df_combined = pd.merge(df_fija, df_portabilidad_filtered, on='año', how='outer').sort_values('año')

    #rellenar los nan con ceros para evitar errores en el modelo
    df_combined['total_fija'] = df_combined['total_fija'].fillna(0)
    df_combined['total_portabilidad'] = df_combined['total_portabilidad'].fillna(0)

    years_future = np.array(range(df_combined['año'].max() + 1, 2027)).reshape(-1, 1)

    #proyeccion para total_fija usando un modelo lineal
    X_fija = df_combined['año'].values.reshape(-1, 1)
    model_fija = LinearRegression().fit(X_fija, df_combined['total_fija'].values)
    future_fija = model_fija.predict(years_future)

    #proyeccion para total_portabilidad usando un modelo lineal
    model_portabilidad = LinearRegression().fit(X_fija, df_combined['total_portabilidad'].values)
    future_portabilidad = model_portabilidad.predict(years_future)

    #agregar las proyecciones a df_combined
    df_future = pd.DataFrame({'año': years_future.flatten(), 'total_fija': future_fija, 'total_portabilidad': future_portabilidad})
    df_combined = pd.concat([df_combined, df_future])

    #crear el grafico de lineas
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_combined['año'], 
        y=df_combined['total_fija'], 
        mode='lines+markers', 
        name='Banda Ancha Fija',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df_combined['año'], 
        y=df_combined['total_portabilidad'], 
        mode='lines+markers', 
        name='Portabilidad',
        line=dict(color='orange')
    ))

    #personalizar el grafico
    fig.update_layout(
        title='Comparación Banda Ancha Fija vs Portabilidad',
        xaxis_title='Año',
        yaxis_title='Total de Accesos',
        hovermode='x',
        template='plotly_white'
    )
    return fig


#callback para el grafico de pastel (portabilidad)
@app.callback(
    Output('pie-chart-portabilidad', 'figure'),
    Input('pie-chart-portabilidad', 'id')  #dummy input, solo para actualizar el grafico
)
def update_pie_chart_portabilidad(_):
    #elegir la ultima fila de datos
    last_row = df_portabilidad.iloc[0]
    df_size = last_row[2:-1].reset_index()
    df_size.columns = ['empresa', 'abonos']

    #definir colores de empresa
    color_map = {
        'personal': 'blue',
        'claro': 'orange',
        'movistar': 'red'
    }

    #crear grafico de pastel
    fig = px.pie(df_size, names='empresa', values='abonos', title='Tamaño de Abonados por Empresa (Última Medición)', hole=0.3, color_discrete_map=color_map)
    
    return fig



#agregar la columna 'region' al dataframe utilizando el diccionario
df_tecnologia['region'] = df_tecnologia['provincia'].map(provincias_regiones)

#callback para actualizar el grafico de fibra optica por region
@app.callback(
    Output('line-fibra-optica', 'figure'),
    Input('dropdown-region-fibra', 'value')
)
def update_line_fibra(selected_region):
    #filtrar las columnas necesarias y eliminar filas con el ano 2024
    df_fibra = df_tecnologia[['año', 'region', 'fibra_óptica']]
    df_fibra = df_fibra[df_fibra['año'] != 2024]

    #agrupar por ano y region, sumando los accesos de fibra optica
    df_fibra_yearly = df_fibra.groupby(['año', 'region']).sum().reset_index()

    #crear proyecciones de crecimiento para 2024-2026 usando la linea de tendencia promedio
    regiones = df_fibra_yearly['region'].unique()
    años_proyeccion = [2024, 2025, 2026]
    nuevos_datos = []

    #iterar sobre cada region para calcular el crecimiento promedio y proyectar datos futuros
    for region in regiones:
        #filtrar datos por region
        df_region = df_fibra_yearly[df_fibra_yearly['region'] == region]
        
        #extraer los anos y valores de accesos
        X = df_region['año'].values.reshape(-1, 1)
        y = df_region['fibra_óptica'].values

        #ajustar un modelo de regresion lineal
        model = LinearRegression()
        model.fit(X, y)

        #predecir los valores futuros para los anos de proyeccion
        for año in años_proyeccion:
            prediccion = model.predict(np.array([[año]]))
            nuevos_datos.append({'año': año, 'region': region, 'fibra_óptica': prediccion[0]})

    #crear un dataframe con los datos proyectados
    df_proyeccion = pd.DataFrame(nuevos_datos)

    #concatenar los datos originales y los proyectados
    df_fibra_total = pd.concat([df_fibra_yearly, df_proyeccion])

    #filtrar por region seleccionada
    if selected_region != 'Todos':
        df_fibra_total = df_fibra_total[df_fibra_total['region'] == selected_region]

    #graficar la tendencia de fibra optica con plotly
    fig = go.Figure()

    #agregar las lineas para cada region o la region seleccionada
    regiones_a_graficar = [selected_region] if selected_region != 'Todos' else regiones
    for region in regiones_a_graficar:
        df_region = df_fibra_total[df_fibra_total['region'] == region]
        fig.add_trace(go.Scatter(
            x=df_region['año'],
            y=df_region['fibra_óptica'],
            mode='lines+markers',
            name=region,
            line_shape='spline'
        ))

    #configurar el diseno del grafico
    fig.update_layout(
        title='Crecimiento del Acceso a la Fibra Óptica por Región',
        xaxis_title='Año',
        yaxis_title='Accesos de Fibra Óptica',
        hovermode='x',
        template='plotly_white'
    )

    return fig



#ejecutar la aplicacion
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)
