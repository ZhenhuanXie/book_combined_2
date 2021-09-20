#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the code for the Jupyter widgets. It is not required
for the model framework. The widgets are purely for decorative purposes.
"""

#######################################################
#                    Dependencies                     #
#######################################################

from ipywidgets import widgets, Layout, Button, HBox, VBox, interactive
from IPython.core.display import display
from IPython.display import clear_output, Markdown, Latex
from IPython.display import Javascript
import numpy as np
import os
try:
    import plotly.graph_objs as go
    from plotly.tools import make_subplots
    from plotly.offline import init_notebook_mode, iplot
except ImportError:
    print("Installing plotly. This may take a while.")
    from pip._internal import main as pipmain
    pipmain(['install', 'plotly'])
    import plotly.graph_objs as go
    from plotly.tools import make_subplots
    from plotly.offline import init_notebook_mode, iplot

# Define global parameters for parameter checks
params_pass = False
model_solved = False

#######################################################
#          Jupyter widgets for user inputs            #
#######################################################

## This section creates the widgets that will be diplayed and used by the user
## to input parameter values.

style_mini = {'description_width': '5px'}
style_short = {'description_width': '100px'}
style_med = {'description_width': '200px'}
style_long = {'description_width': '200px'}

layout_mini =Layout(width='18.75%')
layout_50 =Layout(width='50%')
layout_med =Layout(width='70%')

widget_layout = Layout(width = '100%')

productivity = widgets.BoundedFloatText( ## fraction of new borns
    value=0.0355,
    min = 0,
    max = 2,
    step=0.0001,
    disabled=False,
    description = 'Productivity $\mathbf{a}$',
    style=style_med,
    layout = Layout(width='70%')
)
alpha_k = widgets.BoundedFloatText( ## death rate
    value=0.025,
    min = 0,
    max = 1,
    step=0.0001,
    disabled=False,
    description = r'Depreciation $\alpha_k$',
    style = style_med,
    layout = Layout(width='70%')
)
phi1 = widgets.BoundedFloatText( ## death rate
    value=0.0125,
    min = 0.00001,
    max = 10000,
    step=0.00001,
    disabled=False,
    description = '$\phi_1$',
    style = style_med,
    layout = Layout(width='70%')
)
phi2 = widgets.BoundedFloatText( ## death rate
    value=400,
    min = 0.00001,
    max = 10000,
    step=0.00001,
    disabled=False,
    description = '$\phi_2$',
    style = style_med,
    layout = Layout(width='70%')
)
beta_1 = widgets.BoundedFloatText( ## death rate
    value=0.014,
    min = 0,
    max = 1,
    step=0.0001,
    disabled=False,
    description = r'Technology shock ($\beta_1$)',
    style = style_med,
    layout = Layout(width='70%')
)
beta_2 = widgets.BoundedFloatText( ## death rate
    value=0.0022,
    min = 0,
    max = 10,
    step=0.0001,
    disabled=False,
    description = r'Preferences shock ($\beta_2$)',
    style = style_med,
    layout = Layout(width='70%')
)
delta = widgets.BoundedFloatText( ## death rate
    value=0.005,
    min = 0,
    max = 10,
    step=0.0001,
    disabled=False,
    description = r'Rate of Time Preference $\delta$',
    style = style_med,
    layout = Layout(width='70%')
)
rhos = widgets.Text( ## death rate
    value="0.667, 1.0000001, 1.5",
    disabled=False,
    description = r'Inverse IES $\rho$',
    style = style_med,
    layout = Layout(width='70%')
)

B11 = widgets.BoundedFloatText( ## cov11
    value= 0.011,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
B12 = widgets.BoundedFloatText( ## cov11
    value= 0.025,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
B13 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
B21 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
B22 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
B23 = widgets.BoundedFloatText( ## cov11
    value= 0.119,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigk1 = widgets.BoundedFloatText( ## cov11
    value= 0.477,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigk2 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigk3 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)

U_k1 = widgets.BoundedFloatText( ## cov11
    value= 1.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
U_k2 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)

sigd1 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigd2 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
sigd3 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)

U_d1 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)
U_d2 = widgets.BoundedFloatText( ## cov11
    value= 0.0,
    step= 0.0001,
    min = -10,
    max = 10,
    disabled=False,
    style = style_mini,
    layout = layout_mini
)

shock = widgets.Dropdown(
    options = {'1', '2', '3'},
    value = '1',
    description='Shock index:',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

order = widgets.Dropdown(
    options = {'1', '2'},
    value = '1',
    description='Solution order:',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

gamma = widgets.BoundedFloatText(
    value=10.,
    min = 1.,
    max = 20,
    step=0.01,
    disabled=False,
    description = r'$\gamma \in (1,20]$',
    style=style_med,
    layout = Layout(width='70%')
)

sim_var_names_no_habit = ['V/C', 'R/C', 'C/K', 'I/K', \
                            'K_{t+1}/K_t', 'Z1', 'Z2']
sim_var_names_internal_habit = ['V/H', 'R/H', 'U/H', 'C/K', 'I/K', 'MH/MU','MC/MU',\
                                'H/K','K_{t+1}/K_t', 'Z1', 'Z2']


slider_var = widgets.Dropdown(
    options = {'γ','α','χ','ϵ'},
    value = 'γ',
    description='Parameter to slide over:',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

#slider_var_γ_only = widgets.Dropdown(
#    options = {'γ'},
#    value = 'γ',
#    description='Parameter to slide over:',
#    disabled=False,
#    style = {'description_width': '180px'},
#    layout = Layout(width='70%')
#)

slider_min= widgets.BoundedFloatText(
    value=2.,
    disabled=False,
    description = 'Min',
    style=style_med,
    layout = Layout(width='70%')
)

slider_max= widgets.BoundedFloatText(
    value=10.,
    disabled=False,
    description = 'Max',
    style=style_med,
    layout = Layout(width='70%')
)

slider_step= widgets.BoundedFloatText(
    value=1.,
    disabled=False,
    description = 'Step',
    style=style_med,
    layout = Layout(width='70%')
)

habit = widgets.Dropdown(
    options = [('No Habit',1), ('Habit Internalized',3)],
    value = 3,
    description='Habit type:',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

chi = widgets.BoundedFloatText( ## cov11
    value= 0.9,
    step= 0.05,
    min = 0,
    max = .9999,
    description=r'$\chi \in (0,1)$',
    disabled=False,
    style = style_med,
    layout = layout_med
)

alpha = widgets.BoundedFloatText( ## cov11
    value= 0.9,
    step= 0.0001,
    min = 0,
    max = .9999,
    description=r'$\alpha \in (0,1)$',
    disabled=False,
    style = style_med,
    layout = layout_med
)

epsilon = widgets.BoundedFloatText( ## cov11
    value= 3,
    step= 0.05,
    min = 0,
    max = 1000,
    description=r'$\epsilon \in (0, 20)$',
    disabled=False,
    style = style_med,
    layout = layout_med
)

#def displayHabit(habit):
#    ## This function displays the box to input households productivity
#    ## if hosueholds are allowed to hold capital.
#    #if habit == 1:
#    #    chi.layout.display = 'none'
#    #    alpha.layout.display = 'none'
#    #    epsilon.layout.display = 'none'
#    #    chi.value = 0.9
#    #    alpha.value = 0.9
#    #    epsilon.value = 10
#    #else:
#    chi.layout.display = None
#    alpha.layout.display = None
#    epsilon.layout.display = None
#    chi.value = 0.9
#    alpha.value = 0.9
#    epsilon.value = 10
#    display(chi)
#    display(alpha)
#    display(epsilon)

#def displayHabit(habit):
#    ## This function displays the box to input households productivity
#    ## if hosueholds are allowed to hold capital.
#    if habit == 1:
#        chi.layout.visibility = 'hidden'
#        alpha.layout.visibility = 'hidden'
#        epsilon.layout.visibility = 'hidden'
#        chi.value = 0.9
#        alpha.value = 0.9
#        epsilon.value = 3
#        display(chi)
#        display(alpha)
#        display(epsilon)
#    else:
#        chi.layout.visibility = 'visible'
#        alpha.layout.visibility = 'visible'
#        epsilon.layout.visibility = 'visible'
#        chi.value = 0.9
#        alpha.value = 0.9
#        epsilon.value = 3
#        display(chi)
#        display(alpha)
#        display(epsilon)
        
#def displaySlider(habit):
#    ## This function displays the box to input households productivity
#    ## if hosueholds are allowed to hold capital.
#    if habit == 1:
#        slider_var.layout.display = 'none'
#        slider_var.value = 'γ'
#        slider_var_γ_only.layout.display = None
#        slider_var_γ_only.value = 'γ'
#        display(slider_var_γ_only)
#    else:
#        slider_var_γ_only.layout.display = 'none'
#        slider_var_γ_only.value = 'γ'
#        slider_var.layout.display = None
#        slider_var.value = 'γ'
#        display(slider_var)

#habitOut = widgets.interactive_output(displayHabit, {'habit': habit})
#sliderOut = widgets.interactive_output(displaySlider, {'habit': habit})

timeHorizon = widgets.BoundedIntText( ## death rate
    value=100,
    min = 10,
    max = 2000,
    step=10,
    disabled=False,
    description = 'Time Horizon (quarters)',
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

plotName = widgets.Text(
    value='Stochastic Growth',
    placeholder='Stochastic Growth',
    description='Plot Title',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

conf_int = widgets.BoundedFloatText(
    value= .9,
    step= 0.0001,
    min = 0,
    max = .9999,
    description='Risk Price Confidence Interval',
    disabled=False,
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

checkParams = widgets.Button(
    description='Update parameters',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

runSim = widgets.Button(
    description='Run simulation',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

checkParams2 = widgets.Button(
    description='Update parameters',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

runSlider = widgets.Button(
    description='Run models',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

#displayPlotPanel = widgets.Button(
#    description='Display plot',
#    disabled=False,
#    button_style='', # 'success', 'info', 'warning', 'danger' or ''
#)


box_layout       = Layout(width='100%', flex_flow = 'row')#, justify_content='space-between')
box_layout_wide  = Layout(width='100%', justify_content='space-between')
box_layout_small = Layout(width='10%')

Economy_box = VBox([widgets.Label(value="Economy"), productivity, alpha_k], \
layout = Layout(width='90%'))
Adjustment_box = VBox([widgets.Label(value="Adjustment Costs"), phi1, phi2], \
layout = Layout(width='90%'))
Preferences_box = VBox([widgets.Label(value="Preference parameters"), rhos, delta], \
layout = Layout(width='90%'))
Persistence_box = VBox([widgets.Label(value="Shock persistence"), beta_1, beta_2], \
layout = Layout(width='90%'))

B_box1 = HBox([B11, B12, B13], layout = Layout(width='100%'))
B_box2 = HBox([B21, B22, B23], layout = Layout(width='100%'))
B_box = VBox([widgets.Label(value="B matrix"),B_box1, B_box2], \
             layout = Layout(width='100%'))

sigk_box1 = HBox([sigk1, sigk2, sigk3], layout = Layout(width='100%'))
sigk_box = VBox([widgets.Label(value=r"$\sigma_k$"), sigk_box1], layout = Layout(width='90%'))
U_k_box1 = HBox([U_k1, U_k2], layout = Layout(width='100%'))
U_k_box = VBox([widgets.Label(value=r"$U_k$"), U_k_box1], layout = Layout(width='90%'))
sigd_box1 = HBox([sigd1, sigd2, sigd3], layout = Layout(width='100%'))
sigd_box = VBox([widgets.Label(value=r"$\sigma_d$"), sigd_box1], layout = Layout(width='90%'))
U_d_box1 = HBox([U_d1, U_d2], layout = Layout(width='100%'))
U_d_box = VBox([widgets.Label(value=r"$U_d$"), U_d_box1], layout = Layout(width='90%'))

#habit_box = VBox([widgets.Label(value="Habit"), habit, gamma, habitOut], \
#layout = Layout(width='90%'))
habit_box = VBox([widgets.Label(value="Habit"), habit, gamma, chi, alpha, epsilon], \
layout = Layout(width='90%'))
order_box = VBox([widgets.Label(value="Solution details"), order, timeHorizon], layout = Layout(width='90%'))
slider_box = VBox([widgets.Label(value="Slider setting"), slider_var,slider_min,slider_max,slider_step], layout = Layout(width='90%'))

tech_shock_box = VBox([sigk_box, U_k_box], layout = Layout(width='100%'))
pref_shock_box = VBox([sigd_box, U_d_box], layout = Layout(width='100%'))

Selector_box = VBox([widgets.Label(value="Graph parameters"), shock, conf_int, plotName], \
                    layout = Layout(width='90%'))


simulate_no_habit = widgets.SelectMultiple( options = sim_var_names_no_habit,
    value = ['C/K'],
    rows = len(sim_var_names_no_habit),
    disabled = False
)
simulate_box_no_habit = VBox([widgets.Label(value="Select variables to simulate:"),simulate_no_habit], layout = Layout(width='100%'))

simulate_internal_habit = widgets.SelectMultiple( options = sim_var_names_internal_habit,
    value = ['C/K'],
    rows = len(sim_var_names_internal_habit),
    disabled = False
)
simulate_box_internal_habit = VBox([widgets.Label(value="Select variables to simulate:"),simulate_internal_habit], layout = Layout(width='100%'))


habit_box_layout = Layout(width='56%', flex_flow = 'row')
line1      = HBox([Economy_box, Adjustment_box], layout = box_layout)
line2      = HBox([Preferences_box, Persistence_box], layout = box_layout)
line3      = HBox([B_box, tech_shock_box], layout = box_layout)
line4      = HBox([habit_box, pref_shock_box], layout = box_layout)
line5      = HBox([Selector_box, order_box], layout = box_layout)

fixed_params_Panel = VBox([line1, line2, line3, line4, line5])
run_box_sim = VBox([widgets.Label(value="Run simulation"), checkParams, runSim], layout = Layout(width='100%'))
run_box_slider = VBox([widgets.Label(value="Run multiple models"), checkParams2, runSlider], layout = Layout(width='100%'))
#run_box_rfr = VBox([widgets.Label(value="Execute Model"), runRFR, displayRFRMoments, displayRFRPanel])

simulate_box_no_habit_run = HBox([simulate_box_no_habit, run_box_sim], layout = Layout(width='100%'))
simulate_box_internal_habit_run = HBox([simulate_box_internal_habit, run_box_sim], layout = Layout(width='100%'))

slider_box_run = HBox([slider_box, run_box_slider], layout = Layout(width='100%'))

#######################################################
#                      Functions                      #
#######################################################

def checkParamsFn(b):
    ## This is the function triggered by the updateParams button. It will
    ## check dictionary params to ensure that adjustment costs are well-specified.
    clear_output() ## clear the output of the existing print-out
    display(Javascript("Jupyter.notebook.execute_cells([2])")) #reload the new parameters
    if habit.value == 1:
        display(simulate_box_no_habit_run) ## after clearing output, re-display buttons
    elif habit.value == 3:
        display(simulate_box_internal_habit_run) ## after clearing output, re-display buttons
    global params_pass
    global model_solved
    model_solved = False
    if phi1.value * phi2.value < .6:
        params_pass = False
        print("Given your parameter values, phi 1 must be greater than {}.".format(round(.6 / phi2.value, 3)))
    else:
        rho_vals = np.array([np.float(r) for r in rhos.value.split(',')])
        rho_min = np.min(rho_vals)
        rho_max = np.max(rho_vals)
        upper_cutoff = (delta.value / (1 - rho_min) + alpha_k.value) / \
                        np.log(1 + phi2.value * productivity.value)
        lower_cutoff = (delta.value / (1 - rho_max) + alpha_k.value) / \
                        np.log(1 + phi2.value * productivity.value)
        lower_cutoff = min(lower_cutoff, 0.6 / phi2.value)
        if rho_min < 1 and rho_max > 1:
            if phi1.value > upper_cutoff:
                params_pass = False
                print("Given your parameter values, phi 1 must be greater than {} and less than {}.".format(round(lower_cutoff, 3), round(upper_cutoff, 3)))
            elif phi1.value < lower_cutoff:
                params_pass = False
                print("Given your parameter values, phi 1 must be greater than {} and less than {}.".format(round(lower_cutoff, 3), round(upper_cutoff, 3)))
            else:
                params_pass = True
                print("Parameter check passed.")
        elif rho_max > 1:
            if phi1.value < lower_cutoff:
                params_pass = False
                print("Given your parameter values, phi 1 must be greater than {}.".format(round(lower_cutoff, 3)))
            else:
                params_pass = True
                print("Parameter check passed.")
        elif rho_max < 1:
            if phi1.value > upper_cutoff:
                params_pass = False
                print("Given your parameter values, phi 1 must be less than {}.".format(round(upper_cutoff, 3)))
            else:
                params_pass = True
                print("Parameter check passed.")
        else:
            params_pass = True
            print("Parameter check passed.")
            
def checkParams2Fn(b):
    ## This is the function triggered by the updateParams button. It will
    ## check dictionary params to ensure that adjustment costs are well-specified.
    clear_output() ## clear the output of the existing print-out
    display(Javascript("Jupyter.notebook.execute_cells([2])")) #reload the new parameters
    display(slider_box_run) ## after clearing output, re-display buttons
    global params_pass
    global model_solved
    model_solved = False
    if phi1.value * phi2.value < .6:
        params_pass = False
        print("Given your parameter values, phi 1 must be greater than {}.".format(round(.6 / phi2.value, 3)))
    else:
        rho_vals = np.array([np.float(r) for r in rhos.value.split(',')])
        rho_min = np.min(rho_vals)
        rho_max = np.max(rho_vals)
        upper_cutoff = (delta.value / (1 - rho_min) + alpha_k.value) / \
                        np.log(1 + phi2.value * productivity.value)
        lower_cutoff = (delta.value / (1 - rho_max) + alpha_k.value) / \
                        np.log(1 + phi2.value * productivity.value)
        lower_cutoff = min(lower_cutoff, 0.6 / phi2.value)
        if rho_min < 1 and rho_max > 1:
            if phi1.value > upper_cutoff:
                params_pass = False
                print("Given your parameter values, phi 1 must be greater than {} and less than {}.".format(round(lower_cutoff, 3), round(upper_cutoff, 3)))
            elif phi1.value < lower_cutoff:
                params_pass = False
                print("Given your parameter values, phi 1 must be greater than {} and less than {}.".format(round(lower_cutoff, 3), round(upper_cutoff, 3)))
            else:
                params_pass = True
                print("Parameter check passed.")
        elif rho_max > 1:
            if phi1.value < lower_cutoff:
                params_pass = False
                print("Given your parameter values, phi 1 must be greater than {}.".format(round(lower_cutoff, 3)))
            else:
                params_pass = True
                print("Parameter check passed.")
        elif rho_max < 1:
            if phi1.value > upper_cutoff:
                params_pass = False
                print("Given your parameter values, phi 1 must be less than {}.".format(round(upper_cutoff, 3)))
            else:
                params_pass = True
                print("Parameter check passed.")
        else:
            params_pass = True
            print("Parameter check passed.")

def runSimFn(b):
    ## This is the function triggered by the runSim button.
    global model_solved
    if params_pass:
        print("Running simulation...")
        display(Javascript("Jupyter.notebook.execute_cells([5, 6])"))
        model_solved = True
    else:
        print("You must update the parameters first.")
        
def runSliderFn(b):
    ## This is the function triggered by the runSim button.
    global model_solved
    if params_pass:
        print("Running models...")
        display(Javascript("Jupyter.notebook.execute_cells([9])"))
        model_solved = True
    else:
        print("You must update the parameters first.")

#def showSSFn(b):
#    if model_solved:
#        print("Showing steady state values.")
#        display(Javascript("Jupyter.notebook.execute_cells([15])"))
#    else:
#        print("You must run the model first.")

#def displayPlotPanelFn(b):
#    if model_solved:
#        print("Showing plots.")
#        display(Javascript("Jupyter.notebook.execute_cells([13])"))
#    else:
#        print("You must run the model first.")
#
#def runRFRFn(b):
#    clear_output() ## clear the output of the existing print-out
#    display(run_box_rfr) ## after clearing output, re-display buttons
#    if model_solved:
#        print("Calculating values.")
#        display(Javascript("Jupyter.notebook.execute_cells([19, 20])"))
#    else:
#        print("You must run the model first.")
#
#def displayRFRMomentsFn(b):
#    print("Showing moment table.")
#    display(Javascript("Jupyter.notebook.execute_cells([21,22])"))
#
#def displayRFRPanelFn(b):
#    print("Showing plots.")
#    display(Javascript("Jupyter.notebook.execute_cells([23])"))

#######################################################
#                 Configure buttons                   #
#######################################################

selectedMoments = []

checkParams.on_click(checkParamsFn)
checkParams2.on_click(checkParams2Fn)
runSim.on_click(runSimFn)
runSlider.on_click(runSliderFn)
#showSS.on_click(showSSFn)
#displayPlotPanel.on_click(displayPlotPanelFn)
#runRFR.on_click(runRFRFn)
#displayRFRMoments.on_click(displayRFRMomentsFn)
#displayRFRPanel.on_click(displayRFRPanelFn)
