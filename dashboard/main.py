import time
import dash
from dash import html
from dash.long_callback import DiskcacheLongCallbackManager
from dash.dependencies import Input, Output
import dash_daq as daq
from dash import dcc
import stress_detection as stress
import sklearn
import numpy as np
from tensorflow import keras
import plotly.express as px
import dash_bootstrap_components as dbc

## Diskcache
import diskcache
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
#[dbc.themes.BOOTSTRAP]
#[dbc.themes.CYBORG]
#"https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css"
app = dash.Dash(__name__, long_callback_manager=long_callback_manager, external_stylesheets=external_stylesheets)

participants_list = ['p0', 'p1', 'p3',]

red_button_style = {'background-color': 'red',
                    'color': 'white',
                    'height': '50px',
                    'width': '100px',
                    'margin-top': '50px',
                    'margin-left': '50px'}

app.layout = html.Div(
    [
        html.Br(),

        html.H3("STRESS DETECTION DASHBOARD", 
                style={"marginTop": 5, 
                       "marginLeft": "40px",
                       'color':'black',
                       }), #'text-align': 'center',"border":"2px black solid"
        
        html.H6("Select participant:", style={'text-align': 'left'},),
        html.Div([
            html.Div([
                dcc.Dropdown(id="select_pid",
                    #options=participants_list,
                    options = [
                        {'label': 'Participant 1', 'value':'p0'},
                        {'label': 'Participant 2', 'value':'p1'},
                        {'label': 'Participant 3', 'value':'p2', 'disabled': True},
                    ],
                    multi=False,
                    value='p0',
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
    output=[Output("stress_meter", "value"), 
            Output("final_output", "children")],
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
    progress=[Output("stress_meter", "value"),Output("label_output","children"),],
    prevent_initial_call=True,
)

def detect_stress(set_progress, pid_selected, detect_action):
    display = ""
    meter = 0
    container = ""
    meter_val = 0
    zero, one, two = 0, 0, 0

    if(detect_action == 1):

        #print(pid_selected)
        
        data = {}
        X_test = []
        data = stress.read_processed_data(pid_selected)
        print("data read")

        X_test = stress.prepare_data(data)
        print("data prepared")
        
        model_path = 'dashboard\\my_model.h5'
        model = keras.models.load_model(model_path)
        print("model loaded")
        
        preds = model.predict(X_test)

        labels = np.argmax(preds,axis=1)

        res = np.max(preds, axis = 1) 
        print("res predicted")

        for label in labels:
            time.sleep(0.01)
            #print(label)
            if label == 0:
                zero = zero + 1
                display = "Processing..Zero.."
                meter = 20
            elif label == 1:
                one = one + 1
                display = "Processing..One.."
                meter = 50
            elif label == 2:
                two = two + 1
                display = "Processing..Two.."
                meter = 80
            set_progress((meter,display))
        
        if (zero >= one) and (zero >= two):
            meter_val = 20
            container = "You are cool as a cucumber!"
        elif (one >= zero) and (one >= two):
            meter_val = 50
            container = "Live more, stress less!"
        else:
            meter_val = 80
            container = "Give yourself a break. Don't stress too much!"

    return meter_val, container

@app.callback([Output('graph_eda', 'figure'), 
               Output('graph_bvp', 'figure'),],
              [Input('select_pid', 'value'),
               Input('detect_button', 'n_clicks'),])

def set_graphs(pid_selected, detect_action):

    #if(detect_action == 1):
        df_eda, df_bvp = stress.read_raw_data(pid_selected)
    
        fig_eda = px.line(df_eda, x='time', y='eda', title='Your EDA data')
        type(fig_eda)
        fig_bvp = px.line(df_bvp, x='time', y='bvp', title='Your BVP data')

        return fig_eda, fig_bvp

if __name__ == "__main__":
    app.run_server(debug=True)