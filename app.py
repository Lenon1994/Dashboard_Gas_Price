import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash_bootstrap_templates import ThemeSwitchAIO #Usada para mudar o tema


# ========= App ============== #
FONT_AWESOME = ["https://use.fontawesome.com/releases/v5.10.2/css/all.css"]
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.4/dbc.min.css"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc_css])
app.scripts.config.serve_locally = True
server = app.server

# ========== Styles ============ #

template_theme1 = "flatly"
template_theme2 = "vapor"
url_theme1 = dbc.themes.FLATLY
url_theme2 = dbc.themes.VAPOR
tab_card = {"height": "100%"}
#Configuração básica para todos os gráficos
main_config = {
    "hovermode": "x unified",
    "legend": {"yanchor": "top",
               "y":0.9,
               "xanchor":"left",
               "x":0.1,
               "title": {"text": None},
               "font": {"color": "white"},
               "bgcolor": "rgba(0,0,0,0.5)"
               },
    "margin": {"l":0, "r":0, "t":10, "b":0}
    
}

# ===== Limpando o arquivo====== #
#Local dataset: https://www.kaggle.com/datasets/matheusfreitag/gas-prices-in-brazil
df_main = pd.read_csv(r"assets\data_gas.csv")

#df_main.info()
#Ajustado data
df_main["DATA INICIAL"] = pd.to_datetime(df_main["DATA INICIAL"])
df_main["DATA FINAL"] = pd.to_datetime(df_main["DATA FINAL"])

#Nova coluna Data média
df_main["DATA MEDIA"] = ((df_main["DATA FINAL"] - df_main["DATA INICIAL"] )/2) + df_main["DATA INICIAL"]

#Ordernando o dataframe pela nova colina DATA MEDIA
df_main = df_main.sort_values(by="DATA MEDIA", ascending=True)

#Renomeando colunas
df_main.rename(columns = {"DATA MEDIA" : "DATA"}, inplace= True)
df_main.rename(columns = {"PREÇO MÉDIO REVENDA" : "VALOR REVENDA (R$/L)"}, inplace= True)

#Adicionando coluna ano
df_main["ANO"] = df_main["DATA"].apply(lambda x: str (x.year))

#Filtrando somente gasolina
df_main = df_main[df_main.PRODUTO == "GASOLINA COMUM"]

#resetando o index
df_main = df_main.reset_index()

#Deixando somente as colunas que serão utilizados
df_main.drop(["UNIDADE DE MEDIDA", "COEF DE VARIAÇÃO REVENDA", "COEF DE VARIAÇÃO DISTRIBUIÇÃO", "NÚMERO DE POSTOS PESQUISADOS", "DATA INICIAL", "DATA FINAL", "PREÇO MÁXIMO DISTRIBUIÇÃO", "DESVIO PADRÃO DISTRIBUIÇÃO", "MARGEM MÉDIA REVENDA", "PREÇO MÍNIMO REVENDA", "PREÇO MÁXIMO REVENDA", "PRODUTO", "PREÇO MÉDIO DISTRIBUIÇÃO","DESVIO PADRÃO REVENDA", "PREÇO MÍNIMO DISTRIBUIÇÃO"], inplace=True, axis=1)

#Base final para o dash
df_store = df_main.to_dict()

# =========  Layout  =========== #
app.layout = dbc.Container(children=[
    #armazendo o dataset
    dcc.Store(id="dataset", data=df_store),
    dcc.Store(id="dataset_fixed", data=df_store),
    dcc.Store(id='controller', data={'play': False}),
    
    #Layout Dash
    #icones https://fontawesome.com/
    #Primeira linha
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Análise dos Preços da gasolina no Brasil")
                        ], sm=12),
                    dbc.Row([
                        dbc.Col([
                            ThemeSwitchAIO(aio_id="theme", themes=[url_theme1, url_theme2]),
                            html.H6("Alterar tema")  
                        ]),
                        dbc.Col([
                            html.I(className="fa fa-sort", style={"font-size": "200%"}) #fa fa-filter
                        ], sm=2, align="center"),
                    ], style={"margin-top": "10px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Código", href="https://github.com/Lenon1994/Dashboards_Gas_Prices_python", target="_blank")    
                        ])   
                    ], style={"margin-top": "10px"})
                    ])
                ])
            ], style=tab_card) 
        ], sm=4 , lg=2), #mobile tamanho 4 e tela tamanho grande
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            #Gráfico do resultado máximo e minino
                            html.H4('Máximo e Mínimos'),
                            dcc.Graph(id='static-maxmin', config={"displayModeBar": False, "showTips": False})
                        ])
                    ])
                ])
            ], style=tab_card)
        ], sm=8 , lg=5),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            #Fitro por ano
                            html.H6('Ano de análise:'),
                            dcc.Dropdown(
                                id="select_ano",
                                value=df_main.at[df_main.index[1],"ANO"],
                                clearable= False,
                                className='dbc',
                                options=[
                                    {"label": x, "value": x} for x in df_main.ANO.unique()
                                ]
                            )
                        ],sm=6),
                        dbc.Col([
                            #Fitro por região
                            html.H6('Região da análise:'),
                            dcc.Dropdown(
                                id="select_regiao",
                                value=df_main.at[df_main.index[1],"REGIÃO"],
                                clearable= False,
                                className='dbc',
                                options=[
                                    {"label": x, "value": x} for x in df_main.REGIÃO.unique()
                                ]
                            )
                        ],sm=6)
                    ]),
                    #Grafico horizontal
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='regiaobar_graph', config={"displayModeBar": False, "showTips": False})
                        ], sm=12, md=6),
                        dbc.Col([
                            dcc.Graph(id='estadobar_graph', config={"displayModeBar": False, "showTips": False})
                        ], sm=12, md=6)
                    ], style={'column-gap': '0px'})
                ])
            ], style=tab_card)
        ], sm=12, md=6, lg=5)
    ], className='main_row g-2 my-auto'),
    
# Segunda linha do projeto
    
    dbc.Row([
        #Grafico bignumbers
        dbc.Col([
            #Grafico bignumber 1
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                       dbc.CardBody([
                           dcc.Graph(id='card1_indicadores', config={"displayModeBar": False, "showTips": False}),
                       ]) 
                    ],style=tab_card)
                ])
            ],justify="center", style={"height": "50%"}),
            #Grafico bignumber 2
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                       dbc.CardBody([
                           dcc.Graph(id='card2_indicadores', config={"displayModeBar": False, "showTips": False}),
                       ]) 
                    ],style=tab_card)
                ])
            ],justify="center", style={"height": "50%"})
        ], sm=12, lg=2, style={"height": "100%"}),
    
    
    #Grafico de comparação entre estados
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4('Comparação Direta'),
                    html.H6('Qual o preço é menor em um dado período de tempo?'),
                    dbc.Row([
                        dbc.Col([
                                dcc.Dropdown(
                                id="select_estados1",
                                value=df_main.at[df_main.index[3],"ESTADO"],
                                clearable= False,
                                className='dbc',
                                options=[
                                    {"label": x, "value": x} for x in df_main.ESTADO.unique()
                                ]
                            ),
                        ], sm=10, md=5),
                        dbc.Col([
                                dcc.Dropdown(
                                id="select_estados2",
                                value=df_main.at[df_main.index[1],"ESTADO"],
                                clearable= False,
                                className='dbc',
                                options=[
                                    {"label": x, "value": x} for x in df_main.ESTADO.unique()
                                ]
                            ),
                        ], sm=10, md=5),
                    ], style={'margin-top': '20px'}, justify='center'),
                    dcc.Graph(id='direct_comparison_graph', config={"displayModeBar": False, "showTips": False}),
                    html.P(id='desc_comparison', style={"color": "gray", "font-size": '80%'}),
                ])
            ], style=tab_card)
        ],sm=12, md=6, lg=5),
        
    #Grafico preço X  Estado
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                   html.H4('Preço X Estado'),
                   html.H6('Comparação temporal entre estados') ,
                   dbc.Row([
                       dbc.Col([
                           dcc.Dropdown(
                                id="select_estados0",
                                value=[df_main.at[df_main.index[3],"ESTADO"],df_main.at[df_main.index[13],"ESTADO"], df_main.at[df_main.index[6],"ESTADO"]],
                                clearable= False,
                                className='dbc',
                                multi=True,
                                options=[
                                    {"label": x, "value": x} for x in df_main.ESTADO.unique()
                                ]
                            ),
                       ], sm=10)
                   ]),
                   dbc.Row([
                       dbc.Col([
                           dcc.Graph(id='animation_graph', config={"displayModeBar": False, "showTips": False})
                       ])
                   ])
                   
                ])
            ], style=tab_card)
        ], sm=12, md=6, lg=5),
    ],className='main_row g-2 my-auto'),
        

#Terceira linha do projeto
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.Row([
                    dbc.Col([
                        #Botão da evolução anual
                        dbc.Button([html.I(className="fa - fa-play")], id="play-button", style={'margin-right': "15px"}),
                        dbc.Button([html.I(className="fa - fa-stop")], id="stop-button")
                    ], sm=12, md=1, style={"justify-content": "center", "margin-top": "10px"}),
                    dbc.Col([
                            dcc.RangeSlider(
                            id="rangeslider",
                            marks={int(x): f'{x}' for x in df_main['ANO'].unique()},
                            step=3,
                            min=2004,
                            max=2021,
                            className="dbc",
                            value=[2004,2021],
                            dots=True,
                            pushable=3,
                            tooltip={'always_visible':False, 'placement':'bottom'},
                            )
                    ], sm=12 , md=10, style={"margin-top": "15px"}),
                    #Componente invisivel
                    dcc.Interval(id="interval", interval=2000),
                ], className="g-1", style={"height": "20%", "justify-content":"center"})
            ], style=tab_card)
        ])
    ], className="main_row g-2 my-auto")

], fluid=True, style={'height': '100%'})


# ======== Callbacks ========== #

#Gráfico de máximo e mínimo callback
@app.callback(
    Output('static-maxmin', 'figure'), #Gráfico 
    Input("dataset", 'data'),           #Dataset
    Input(ThemeSwitchAIO.ids.switch("theme"), "value") #Tema
)
#Função para criar o gráfico interativo
def func(data, toggle):
    template = template_theme1 if toggle else template_theme2  #função para alteração do tema do gráfico
    
    dff = pd.DataFrame(data) #Dados
    max = dff.groupby(['ANO']) ['VALOR REVENDA (R$/L)'].max() #Resultados máximo por ano
    min = dff.groupby(['ANO']) ['VALOR REVENDA (R$/L)'].min() #Resultados minino por ano
    
    final_df = pd.concat([max, min], axis=1) 
    final_df.columns = ['Máximo', 'Mínimo']  #Dataset final com as novas colunas
    
    #Gráfico gerado com os resultado
    fig = px.line(final_df, x=final_df.index, y=final_df.columns, template=template)
    
    #Update layout gráfico
    fig.update_layout(main_config, height=150, xaxis_title=None, yaxis_title=None)
    
    return fig

#Grafico horizontal
@app.callback(
    [Output('regiaobar_graph', 'figure'),
     Output('estadobar_graph', 'figure')
    ],
    [Input('dataset_fixed', 'data'),
     Input('select_ano','value'),
     Input('select_regiao','value'),
     Input(ThemeSwitchAIO.ids.switch("theme"), "value")
     ]
)

#função para criar o gráfico horizontal interativo
def graf1(data, ano, regiao, toggle):
    template = template_theme1 if toggle else template_theme2 
    
    df = pd.DataFrame(data)
    df_filtered = df[df.ANO.isin([ano])]
    
    dff_regiao = df_filtered.groupby(['ANO','REGIÃO'])['VALOR REVENDA (R$/L)'].mean().reset_index()
    dff_estado = df_filtered.groupby(['ANO','ESTADO', 'REGIÃO'])['VALOR REVENDA (R$/L)'].mean().reset_index()
    dff_estado = dff_estado[dff_estado.REGIÃO.isin([regiao])]
    
    dff_regiao = dff_regiao.sort_values(by='VALOR REVENDA (R$/L)', ascending=True)
    dff_estado = dff_estado.sort_values(by='VALOR REVENDA (R$/L)', ascending=True)
    
    dff_regiao['VALOR REVENDA (R$/L)'] = dff_regiao['VALOR REVENDA (R$/L)'].round(decimals= 2)
    dff_estado['VALOR REVENDA (R$/L)'] = dff_estado['VALOR REVENDA (R$/L)'].round(decimals= 2)
    
    fig1_text = [f'{x} - R${y}' for x,y in zip(dff_regiao.REGIÃO.unique(), dff_regiao['VALOR REVENDA (R$/L)'].unique())]
    fig2_text = [f'R${y} - {x}' for x,y in zip(dff_estado.ESTADO.unique(), dff_estado['VALOR REVENDA (R$/L)'].unique())]
    
    fig1 = go.Figure(go.Bar(
        x=dff_regiao['VALOR REVENDA (R$/L)'],
        y=dff_regiao['REGIÃO'],
        orientation='h',
        text=fig1_text,
        textposition='auto',
        insidetextanchor='end',
        insidetextfont=dict(family='Times', size=12)
    ))
    
    fig2 = go.Figure(go.Bar(
        x=dff_estado['VALOR REVENDA (R$/L)'],
        y=dff_estado['ESTADO'],
        orientation='h',
        text=fig2_text,
        textposition='auto',
        insidetextanchor='end',
        insidetextfont=dict(family='Times', size=12)
    ))
    
    
    fig1.update_layout(main_config, yaxis={'showticklabels': False}, height=140, template=template)
    fig2.update_layout(main_config, yaxis={'showticklabels': False}, height=140, template=template)

    fig1.update_layout(xaxis_range=[dff_regiao['VALOR REVENDA (R$/L)'].max(), dff_regiao['VALOR REVENDA (R$/L)'].min()- 0.15])
    fig2.update_layout(xaxis_range=[dff_estado['VALOR REVENDA (R$/L)'].min() - 0.15, dff_estado['VALOR REVENDA (R$/L)'].max()])
    
    return [fig1, fig2]


#Grafico Preço X Estado
@app.callback(
    Output("animation_graph", "figure"),
    Input("dataset", "data"),
    Input("select_estados0", "value"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")

)

#Função para criar o gráfico Preço X estado
def animation(data, estados, toggle):
    template = template_theme1 if toggle else template_theme2  #função para alteração do tema do gráfico
    
    dff = pd.DataFrame(data)
    mask = dff.ESTADO.isin(estados)
    fig = px.line(dff[mask], x='DATA', y='VALOR REVENDA (R$/L)', color='ESTADO', template=template)
    
    fig.update_layout(main_config, height=425 , xaxis_title=None)
    
    return fig

#Gráfico de comparação direta
@app.callback(
    [Output('direct_comparison_graph', 'figure'),
    Output('desc_comparison', 'children')],
    [Input('dataset', 'data'),
    Input('select_estados1', 'value'),
    Input('select_estados2', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def func(data, est1, est2, toggle):
    template = template_theme1 if toggle else template_theme2

    dff = pd.DataFrame(data)
    df1 = dff[dff.ESTADO.isin([est1])]
    df2 = dff[dff.ESTADO.isin([est2])]
    df_final = pd.DataFrame()
    
    df_estado1 = df1.groupby(pd.PeriodIndex(df1['DATA'], freq="M"))['VALOR REVENDA (R$/L)'].mean().reset_index()
    df_estado2 = df2.groupby(pd.PeriodIndex(df2['DATA'], freq="M"))['VALOR REVENDA (R$/L)'].mean().reset_index()

    df_estado1['DATA'] = pd.PeriodIndex(df_estado1['DATA'], freq="M")
    df_estado2['DATA'] = pd.PeriodIndex(df_estado2['DATA'], freq="M")

    df_final['DATA'] = df_estado1['DATA'].astype('datetime64[ns]')
    df_final['VALOR REVENDA (R$/L)'] = df_estado1['VALOR REVENDA (R$/L)']-df_estado2['VALOR REVENDA (R$/L)']
    
    fig = go.Figure()
    # Toda linha
    fig.add_scattergl(name=est1, x=df_final['DATA'], y=df_final['VALOR REVENDA (R$/L)'])
    # Abaixo de zero
    fig.add_scattergl(name=est2, x=df_final['DATA'], y=df_final['VALOR REVENDA (R$/L)'].where(df_final['VALOR REVENDA (R$/L)'] > 0.00000))

    # Updates
    fig.update_layout(main_config, height=350, template=template)
    fig.update_yaxes(range = [-0.7,0.7])

    # Annotations pra mostrar quem é o mais barato
    fig.add_annotation(text=f'{est2} mais barato',
        xref="paper", yref="paper",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#ffffff"
            ),
        align="center", bgcolor="rgba(0,0,0,0.5)", opacity=0.8,
        x=0.1, y=0.75, showarrow=False)

    fig.add_annotation(text=f'{est1} mais barato',
        xref="paper", yref="paper",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#ffffff"
            ),
        align="center", bgcolor="rgba(0,0,0,0.5)", opacity=0.8,
        x=0.1, y=0.25, showarrow=False) 

    # Definindo o texto
    text = f"Comparando {est1} e {est2}. Se a linha estiver acima do eixo X, {est2} tinha menor preço, do contrário, {est1} tinha um valor inferior"
    return [fig, text]
                       
    
#Indicador 1 (Bignumber)
@app.callback(
    Output("card1_indicadores", "figure"),
    [Input('dataset', 'data'),
     Input('select_estados1','value'),
     Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)

#Função para criar o gráfico do indicador 1 (bignumber) interativo
def card2(data, estado, toggle):
    template = template_theme1 if toggle else template_theme2  #função para alteração do tema do gráfico
    
    dff = pd.DataFrame(data)
    df_final = dff[dff.ESTADO.isin([estado])]
    
    data1 = str(int(dff.ANO.min())-1)
    data2 = dff.ANO.max()
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        title = {"text": f"<span style='size:60%'>{estado}</span><br><span style='font-size:0.7em'>{data1} - {data2}</span>"},
        value = df_final.at[df_final.index[-1], 'VALOR REVENDA (R$/L)'],
        number = {'prefix': "R$", 'valueformat': '.2f'},
        delta = {'relative': True, 'valueformat': '.1%', 'reference': df_final.at[df_final.index[0],'VALOR REVENDA (R$/L)']}
    ))
    
    fig.update_layout(main_config, height=250, template=template)
    
    return fig

#Indicador 2 (Bignumber)
@app.callback(
    Output("card2_indicadores", "figure"),
    [Input('dataset', 'data'),
     Input('select_estados2','value'),
     Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)

#Função para criar o gráfico do indicador 2 (bignumber) interativo
def card2(data, estado, toggle):
    template = template_theme1 if toggle else template_theme2  #função para alteração do tema do gráfico
    
    dff = pd.DataFrame(data)
    df_final = dff[dff.ESTADO.isin([estado])]
    
    data1 = str(int(dff.ANO.min())-1)
    data2 = dff.ANO.max()
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "number+delta",
        title = {"text": f"<span style='size:60%'>{estado}</span><br><span style='font-size:0.7em'>{data1} - {data2}</span>"},
        value = df_final.at[df_final.index[-1], 'VALOR REVENDA (R$/L)'],
        number = {'prefix': "R$", 'valueformat': '.2f'},
        delta = {'relative': True, 'valueformat': '.1%', 'reference': df_final.at[df_final.index[0],'VALOR REVENDA (R$/L)']}
    ))
    
    fig.update_layout(main_config, height=250, template=template)
    
    return fig


# Rangeslider
@app.callback(
    Output('dataset', 'data'),
    [Input('rangeslider', 'value'),
    Input('dataset_fixed', 'data')], prevent_initial_call=True
)
def range_slider(range, data):
    dff = pd.DataFrame(data)
    dff = dff[(dff['ANO'] >= f'{range[0]}-01-01') & (dff['ANO'] <= f'{range[1]}-31-12')]
    data = dff.to_dict()

    return data

# Criando a animação do rangeslider
@app.callback(
    Output('rangeslider', 'value'),
    Output('controller', 'data'), 

    Input('interval', 'n_intervals'),
    Input('play-button', 'n_clicks'),
    Input('stop-button', 'n_clicks'),

    State('rangeslider', 'value'), 
    State('controller', 'data'), 
    prevent_initial_callbacks = True)
def controller(n_intervals, play, stop, rangeslider, controller):
    trigg = dash.callback_context.triggered[0]["prop_id"]

    if ('play-button' in trigg and not controller["play"]):
        if not controller["play"]:
            controller["play"] = True
            rangeslider[1] = 2007
        
    elif 'stop-button' in trigg:
        if controller["play"]:
            controller["play"] = False

    if controller["play"]:
        if rangeslider[1] == 2021:
            controller['play'] = False
        rangeslider[1] += 1 if rangeslider[1] < 2021 else 0
    
    return rangeslider, controller


# Run server
if __name__ == "__main__":
    app.run_server(debug=False)
