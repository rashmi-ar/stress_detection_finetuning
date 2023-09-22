import time
import dash
from dash import html
from dash.long_callback import DiskcacheLongCallbackManager
from dash.dependencies import Input, Output
import dash_daq as daq
from dash import dcc
import stress_level_detection as stress
import sklearn
import numpy as np
from tensorflow import keras
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd

## Diskcache
import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
#external_stylesheets = [dbc.themes.BOOTSTRAP]
#[dbc.themes.BOOTSTRAP]
#[dbc.themes.CYBORG]
#"https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css"
app = dash.Dash(__name__, long_callback_manager=long_callback_manager, external_stylesheets=external_stylesheets)

red_button_style = {'background-color': 'red',
                    'color': 'white',
                    'height': '50px',
                    'width': '100px',
                    'margin-top': '50px',
                    'margin-left': '50px'}


app.layout = html.Div(
    [
        html.Br(),

        html.H2("STRESS DETECTION DASHBOARD", 
                style={"marginTop": 5, 
                       "marginLeft": "40px",
                       'color':'black',
                       'text-align': 'center',
                       "border":"2px black solid",
                       }), 
        
        html.H6("Select participant:", style={'text-align': 'left'},),
        html.Div([
            html.Div([
                dcc.Dropdown(id="select_pid",
                    #options=participants_list,
                    options = [
                        {'label': 'Participant 0', 'value':'p00'},
                        {'label': 'Participant 1', 'value':'p01'},
                        {'label': 'Participant 2', 'value':'p02', 'disabled': True},
                        {'label': 'Participant 3', 'value':'p03'},
                        {'label': 'Participant 4', 'value':'p04'},
                        {'label': 'Participant 5', 'value':'p05'},
                        {'label': 'Participant 6', 'value':'p06'},
                        {'label': 'Participant 7', 'value':'p07'},
                        {'label': 'Participant 8', 'value':'p08'},
                        {'label': 'Participant 9', 'value':'p09'},
                        {'label': 'Participant 10', 'value':'p10'},
                        {'label': 'Participant 11', 'value':'p11'},
                        {'label': 'Participant 12', 'value':'p12'},
                        {'label': 'Participant 13', 'value':'p13'},
                        {'label': 'Participant 14', 'value':'p14'},
                        {'label': 'Participant 15', 'value':'p15'},
                        {'label': 'Participant 16', 'value':'p16'},
                        {'label': 'Participant 17', 'value':'p17'},
                        {'label': 'Participant 18', 'value':'p18', 'disabled': True},
                        {'label': 'Participant 19', 'value':'p19'},
                        {'label': 'Participant 20', 'value':'p20', 'disabled': True},
                        {'label': 'Participant 21', 'value':'p21'},
                        {'label': 'Participant 22', 'value':'p22'},
                        {'label': 'Participant 23', 'value':'p23'},
                        {'label': 'Participant 24', 'value':'p24'},
                    ],
                    multi=False,
                    value='p00',
                    style={'width': "60%", 'align-items': 'center'},
                    #placeholder="Select participant",
                    #searchable=False
                ),
                html.Br(),

                html.Button(id="detect_button", children="Detect Stress", 
                            style={'width': "30%", 'align-items': 'center', 'color':'blue'}),
                html.Button(id="cancel_button_id", children="Cancel",
                            style={'width': "30%", 'align-items': 'center', 'color':'red'}),
            ],className="four columns"),

            html.Div([
                    html.H2(id='final_output', children=[""], style={'color':'green'}),
                    html.H4(id="label_output", children=[""]),
            ],className="six columns"),

        ], className="row"),
        html.Br(),
       
        html.Div([
            html.Div([

                html.Div(
                    daq.Gauge(
                    label="Stress Detection Meter",
                    id='stress_meter',
                    min=0,
                    max=100,
                    color={"gradient":True,"ranges":{"green":[0,40],"yellow":[40,70],"red":[70,100]}},
                    #style={'text-align': 'center'},
                    size=300,
                    showCurrentValue=True,
                    value=0,
                    )
                ),
                
            ], className="four columns"),

            html.Div([
                html.Div(
                    dcc.Graph(id='graph_eda', figure={}, 
                        style={'text-align': 'center', 'width': '90vh', 'height': '30vh'},
                )),
                html.Div(
                    dcc.Graph(id='graph_bvp', figure={}, 
                        style={'text-align': 'center', 'width': '90vh', 'height': '30vh'},
                )),
            ], className="six columns")

        ], className="row"),
        
    ]
)

@app.long_callback(
    output=[Output('final_output',"children")],
    inputs=[Input("select_pid", "value"),
            Input('detect_button', 'n_clicks')],
    running=[
        (Output("detect_button", "disabled"), True, False),
        (Output("cancel_button_id", "disabled"), False, True),
        (
            Output("label_output", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        ),
    ],
    cancel=[Input("cancel_button_id", "n_clicks")],
    progress=[Output("stress_meter", "value"),Output("final_output","children"),],
    prevent_initial_call=True, manager=long_callback_manager,
)

def detect_stress(set_progress, pid_selected, detect_action):
    print(detect_action)
    display = ""
    meter_val = 0

    if(detect_action == 1):
        
        X_test = []
        print("data read")

        X_test = stress.read_test_data(pid_selected)
        print("data prepared")
        
        model_path = 'dashboard\\kfold_fcn_[\''+str(pid_selected)+'\'].h5'

        print(model_path)
        model = keras.models.load_model(model_path)
        print("model loaded")
        
        preds = model.predict(X_test)
        print(preds)

        label = np.max(preds, axis = 1) 
        print(label)
        print("res predicted")
        count = 0

        for i in label:
            time.sleep(0.09)
            meter_val = (1-i)*100
            if meter_val >= 50:
                display = ["Give yourself a break. Don't stress too much!"]
    
            elif meter_val < 50 and meter_val >= 1:
                display = ["You are cool as a cucumber!"]

            set_progress((meter_val,display))

    return [display]

@app.callback([Output('graph_eda', 'figure'), 
               Output('graph_bvp', 'figure'),],
              [Input('select_pid', 'value'),
               Input('detect_button', 'n_clicks'),])

def set_graphs(pid_selected, detect_action):

    #if(detect_action == 1):
        df_eda, df_bvp = stress.read_raw_data(pid_selected)
    
        fig_eda = px.line(df_eda, x='time', y='eda', title='Your EDA data')
        fig_bvp = px.line(df_bvp, x='time', y='bvp', title='Your BVP data')

        return fig_eda, fig_bvp

if __name__ == "__main__":
    app.run_server(debug=True)
