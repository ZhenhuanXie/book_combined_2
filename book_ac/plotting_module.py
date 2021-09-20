import numpy as np
np.set_printoptions(precision=6, suppress=True)
import matplotlib.pyplot as plt
from model_specification import *
from expansion import recursive_expansion
from elasticity import price_elasticity
try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    from plotly.offline import init_notebook_mode, iplot
except ImportError:
    print("Installing plotly. This may take a while.")
    from pip._internal import main as pipmain
    pipmain(['install', 'plotly'])
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    from plotly.offline import init_notebook_mode, iplot

def plot_simulation(rhos, T, ϕ1, ϕ2, α_k, a, δ, A, B, U_k, σ_k, U_d, σ_d,
                 order, γ, habit_val, χ, α, ϵ, selected_index):
    colors = ['blue', 'green', 'red', 'gold', 'cyan', 'magenta', 'yellow', 'salmon', 'grey', 'black']
    titles_no_habit = [r'$\log\frac{V}{C}$', r'$\log\frac{R}{C}$', r'$\log\frac{C}{K}$', r'$\log\frac{I}{K}$', r'$\log\frac{K_{t+1}}{K_{t}}$', r'$Z_1$', r'$Z_2$']
    titles_internal_habit = [r'$\log\frac{V}{H}$', r'$\log\frac{R}{H}$', r'$\log\frac{U}{H}$', r'$\log\frac{C}{K}$', r'$\log\frac{I}{K}$', r'$\log\frac{MH}{MU}$', r'$\log\frac{MC}{MU}$', r'$\log\frac{H}{K}$', r'$\log\frac{K_{t+1}}{K_{t}}$', r'$Z_1$', r'$Z_2$']
    
    fig, axs = plt.subplots((len(selected_index)+1)//2,2, squeeze = False, figsize = (10,4*(len(selected_index)+1)//2), dpi = 200)
    
    solved_models = []
        
    β = np.exp(-δ)
    second_order = False if order == 1 else True
    
                  
    for i, ρ in enumerate(rhos):
        if habit_val == 1:
            eq_cond = eq_cond_no_habit_with_pref_shock
            ss_func = ss_func_no_habit_with_pref_shock
            var_shape = (4,3,3)
            args = (γ, ρ, β, a, ϕ1, ϕ2, α_k, U_k, σ_k, U_d, σ_d, A, B)
            modelSol = recursive_expansion(eq_cond=eq_cond,
                           ss=ss_func,
                           var_shape=var_shape,
                           γ=args[0],
                           second_order=second_order,
                           args=args)
            titles = titles_no_habit
        elif habit_val  == 3:
            eq_cond = eq_cond_internal_habit
            ss_func = ss_func_internal_habit
            var_shape = (7,4,3)
            args = (γ, ρ, β, χ, α, ϵ, a, ϕ1, ϕ2, α_k, U_k, σ_k, A, B)
            modelSol = recursive_expansion(eq_cond=eq_cond,
                           ss=ss_func,
                           var_shape=var_shape,
                           γ=args[0],
                           second_order=second_order,
                           args=args)
            titles = titles_internal_habit
        _, _, n_W = modelSol.var_shape
        if i == 0:
            Ws_1 = np.zeros((T,n_W))
            Ws_2 = np.random.multivariate_normal(np.zeros(n_W), np.eye(n_W), size = T)
        
        sim_result_deterministic = modelSol.simulate(Ws_1)
        sim_result_stochastic = modelSol.simulate(Ws_2)
        
        solved_models.append(modelSol)
        
        for j, index in enumerate(selected_index):
            axs[j//2][j%2].plot(sim_result_deterministic[:,index], color = colors[i], linestyle = 'dashed')
            axs[j//2][j%2].plot(sim_result_stochastic[:,index], color = colors[i], alpha = 0.6, label = r'$\rho = {:.2f}$'.format(ρ))
            axs[j//2][j%2].set_ylabel(titles[index])
            axs[j//2][j%2].set_xlabel('Quarters')
            axs[j//2][j%2].legend(loc = 'lower right')
        if len(selected_index) % 2 != 0:
            axs[-1][-1].set_axis_off()
            
    return fig, axs, solved_models
    
def plot_impulse(rhos, T, ϕ1, ϕ2, α_k, a, δ, A, B, U_k, σ_k, U_d, σ_d,
                 order, gamma, slider_varname, habit_val, chi, alpha, epsilon,
                 shock = 1, title = None, confidence_interval = None):
    """
    Given a set of parameters, computes and displays the impulse responses of
    consumption, capital, the consumption-investment ratio, along with the
    shock price elacticities.

    Input
    ==========
    Note that the values of delta, phi, A, and a_k are specified within the code
    and are only used for the empirical_method = 0 or 0.5 specifications (see below).

    rhos:               The set of rho values for which to plot the IRFs.
    gamma:              The risk aversion of the model.
    betaz:              Shock persistence.
    T:                  Number of periods to plot.
    shock:              (1 or 2) Defines which of the two possible shocks to plot.
    empirical method:   Use 0 to use Eberly and Wang parameters and 0.5 for parameters
                        from a low adjustment cost setting. Further cases still under
                        development.
    transform_shocks:   True or False. True to make the rho = 1 response to
                        shock 2 be transitory.
    title:              Title for the image plotted.
    """
    colors = ['blue', 'green', 'red', 'gold', 'cyan', 'magenta', 'yellow', 'salmon', 'grey', 'black']
    mult_fac = len(rhos) // len(colors) + 1
    colors = colors * mult_fac

    Cmin = 0
    Cmax = 0
    Imin = 0
    Imax = 0
    CmKmin = 0
    CmKmax = 0
    ImKmin = 0
    ImKmax = 0
    ϵ_p_c_min = 0
    ϵ_p_c_max = 0
    
    fig = make_subplots(3, 2, print_grid = False, specs=[[{}, {}], [{},{}],  [{'colspan': 2}, None]])

    # Update slider information
    if slider_varname == 'γ':
        slider_vars = gamma
    elif slider_varname == 'χ':
        slider_vars = chi
    elif slider_varname == 'α':
        slider_vars = alpha
    elif slider_varname == 'ϵ':
        slider_vars = epsilon
        
    solved_models = []
    
    β = np.exp(-δ)
    
    for i, r in enumerate(rhos):
        for j, var in enumerate(slider_vars):
            # Update variables
            if slider_varname == 'γ':
                gamma = var
            elif slider_varname == 'χ':
                chi = var
            elif slider_varname == 'α':
                alpha = var
            elif slider_varname == 'ϵ':
                epsilon = var
            # Specify models with/without habit
            if habit_val == 1:
                eq_cond = eq_cond_no_habit_with_pref_shock
                ss_func = ss_func_no_habit_with_pref_shock
                log_SDF_ex = log_SDF_ex_no_habit_with_pref_shock
                var_shape = (4,3,3)
                c_loc = 2
                i_loc = 3
                gk_loc = 4
                args = (gamma, r, β, a, ϕ1, ϕ2, α_k, U_k, σ_k, U_d, σ_d, A, B)

            elif habit_val == 3:
                eq_cond = eq_cond_internal_habit
                ss_func = ss_func_internal_habit
                log_SDF_ex = log_SDF_ex_internal_habit
                var_shape = (7,4,3)
                c_loc = 3
                i_loc = 4
                gk_loc = 8
                args = (gamma, r, β, chi, alpha, epsilon, a, ϕ1, ϕ2, α_k, U_k, σ_k, A, B)
            
            second_order = False if order == 1 else True
            
            modelSol = recursive_expansion(eq_cond=eq_cond,
                               ss=ss_func,
                               var_shape=var_shape,
                               γ=args[0],
                               second_order=second_order,
                               args=args)
            n_Y, n_Z, n_W = modelSol.var_shape
            
            states, controls = modelSol.IRF(T, shock - 1)
            
            CmK_IRF = controls[:,c_loc] * 100
            K_IRF = np.cumsum(states[:,gk_loc-n_Y]) * 100
            C_IRF = CmK_IRF + K_IRF
            ImK_IRF = controls[:, i_loc] * 100
            I_IRF = ImK_IRF + K_IRF
            
            Z2_tp1 = modelSol.Z2_tp1 if second_order else None
            K_growth = modelSol.X_tp1.split()[gk_loc]
            X_growth = modelSol.X_tp1 - modelSol.X_t
            X_growth_list = X_growth.split()
            CmK_growth = X_growth_list[c_loc]
            C_growth = CmK_growth + K_growth
#            ImK_growth = X_growth_list[i_loc]
#            I_growth = ImK_growth + K_growth
            log_SDF = modelSol.approximate_fun(log_SDF_ex, args) + modelSol.log_M
            
            ϵ_p_c = price_elasticity(C_growth, log_SDF, modelSol.Z1_tp1, Z2_tp1, T, shock-1, 0.5).flatten()
                
            if confidence_interval is not None and order == 2:

                ϵ_p_c_lower = price_elasticity(C_growth, log_SDF, modelSol.Z1_tp1, Z2_tp1, T, shock-1, 0.5-confidence_interval/2).flatten()
                ϵ_p_c_upper = price_elasticity(C_growth, log_SDF, modelSol.Z1_tp1, Z2_tp1, T, shock-1, 0.5+confidence_interval/2).flatten()
            
            solved_models.append(modelSol)
            
            Cmin = min(Cmin, np.min(C_IRF) * 1.2)
            Cmax = max(Cmax, np.max(C_IRF) * 1.2)
            Imin = min(Imin, np.min(I_IRF) * 1.2)
            Imax = max(Imax, np.max(I_IRF) * 1.2)
            CmKmin = min(CmKmin, np.min(CmK_IRF) * 1.2)
            CmKmax = max(CmKmax, np.max(CmK_IRF) * 1.2)
            ImKmin = min(ImKmin, np.min(ImK_IRF) * 1.2)
            ImKmax = max(ImKmax, np.max(ImK_IRF) * 1.2)
                
            if confidence_interval is None or order == 1:
                ϵ_p_c_min = min(ϵ_p_c_min, np.min(ϵ_p_c) * 1.2)
                ϵ_p_c_max = max(ϵ_p_c_max, np.max(ϵ_p_c) * 1.2)
                
            else:
                ϵ_p_c_min = min(ϵ_p_c_min, np.min(ϵ_p_c_lower) * 1.2)
                ϵ_p_c_max = max(ϵ_p_c_max, np.max(ϵ_p_c_upper) * 1.2)
            
            fig.add_scatter(y = C_IRF, row = 1, col = 1, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
            fig.add_scatter(y = I_IRF, row = 1, col = 2, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
            fig.add_scatter(y = CmK_IRF, row = 2, col = 1, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
            fig.add_scatter(y = ImK_IRF, row = 2, col = 2, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
                
            fig.add_scatter(y = ϵ_p_c, row = 3, col = 1, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
                            
            
            if confidence_interval is not None and order == 2:
                               
                fig.add_scatter(y = ϵ_p_c_lower, row = 3, col = 1, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
                fig.add_scatter(y = ϵ_p_c_upper, row = 3, col = 1, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])),
                               fill = 'tonexty')
                               
    
    steps = []
    for i in range(len(slider_vars)):
        step = dict(
            method = 'restyle',
            args = ['visible', ['legendonly'] * len(fig.data)],
            label = slider_varname + ' = '+'{}'.format(round(slider_vars[i], 2))
        )
        if confidence_interval is None or order == 1:
            for j in range(5):
                for k in range(len(rhos)):
                    step['args'][1][i * 5 + j + k * len(slider_vars) * 5] = True
        else:
            for j in range(7):
                for k in range(len(rhos)):
                    step['args'][1][i * 7 + j + k * len(slider_vars) * 7] = True
        steps.append(step)

    sliders = [dict(
        steps = steps
    )]

    fig.layout.sliders = sliders
    fig['layout'].update(height=800, width=1000,
                     title=title.format(shock), showlegend = False)

    fig['layout']['xaxis1'].update(range = [0, T])
    fig['layout']['xaxis2'].update(range = [0, T])
    fig['layout']['xaxis3'].update(range = [0, T])
    fig['layout']['xaxis4'].update(range = [0, T])
    fig['layout']['xaxis5'].update(range = [0, T])

    fig['layout']['yaxis1'].update(title=r'$\text{IRF: }\log C$', range = [Cmin, Cmax])
    fig['layout']['yaxis2'].update(title=r'$\text{IRF: }\log I$', range=[Imin, Imax])
    fig['layout']['yaxis3'].update(title=r'$\text{IRF: }\log\frac{C}{K}$', range = [CmKmin, CmKmax])
    fig['layout']['yaxis4'].update(title=r'$\text{IRF: }\log\frac{I}{K}$', range=[ImKmin, ImKmax])
    fig['layout']['yaxis5'].update(title=r'$\text{Price Elasticity: }\log C$', range=[ϵ_p_c_min, ϵ_p_c_max])
    
    return fig, solved_models
