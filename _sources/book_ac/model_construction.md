---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(ac_model)=
# Model construction

(ac_model_preference)=
## Preference: Recursive utility 

(ac_model_preference_nocapital)=
### Without household capital

This is a special case of more general recursive preference, where there's no habits or durable goods. We take a consumption process $\{C_t\}$ as an input into $\{R_t, V_t\}$ processes that we define via backward recursion:

```{math}
    V_t = \left[(1-\beta)C_t^{1-\rho}+\beta R_t^{1-\rho}\right]^{\frac{1}{1-\rho}}
```

```{math}
    R_t = \mathbb{E}\left[V_{t+1}^{1-\gamma} \mid {\mathfrak F}_t\right]^{\frac{1}{1-\gamma}}
```

$\{V_t\}$ is a continuation value process that ranks $\{C_t\}$ processes. The reciprocal of the parameter $\rho$ describes the consumer's attitudes about intertemporal substitution, while the parameter$\gamma$ describes the consumer's attitudes toward risk.

In practice, we are interested in finding a balanced growth path, where some variables grow at constant rates, while others are in a steady state. To this end, it's convenient to use a growing variable to scale others. In this specification, we can use $C$ to scale $V$ and $R$:

```{math}
    \frac{V_t}{C_t} = \left[(1-\beta)+\beta\left(\frac{R_t}{C_t}\right)^{1-\rho}\right]^{\frac{1}{1-\rho}}
```

In the special case of $\rho = 1$,

```{math}
    \frac{V_t}{C_t} = \left(\frac{R_t}{C_t}\right)^{\beta}
```

```{math}
    \frac{R_t}{C_t} = \mathbb{E}\left[\left(\frac{V_{t+1}}{C_t}\right)^{1-\gamma} | {\mathfrak F}_t\right]^{\frac{1}{1-\gamma}}
```

More generally, we can include "preference shock" under this specification:

```{math}
    V_t = \left[(1-\beta)\left(C_t D_t \right)^{1-\rho}+\beta R_t^{1-\rho}\right]^{\frac{1}{1-\rho}}
```

where $\{D_t\}$ is an exogenous preference shifter process, whose dynamic will be introduced later. In this case, we can use $CD$ to scale $V$ and $R$ to obtain a balanced growth version:

```{math}
    \frac{V_t}{C_tD_t} = \left[(1-\beta)+\beta\left(\frac{R_t}{C_tD_t}\right)^{1-\rho}\right]^{\frac{1}{1-\rho}}
```

```{math}
    \frac{R_t}{C_tD_t} = \mathbb{E}\left[\left(\frac{V_{t+1}}{C_tD_t}\right)^{1-\gamma} | {\mathfrak F}_t\right]^{\frac{1}{1-\gamma}}
```

(ac_model_preference_capital)=
### With household capital

This is a more general framework of recursive preference, where we introduce $U_t$ via a CES aggregator of current consumption $C_t$ and a household stock variable $H_t$. $H_t$ can be interpreted either as habits or as durable goods. It will be clear that $H_t$ is a geometrically weighted average of current and past consumptions, and the initial $H_0$. 

Now, $H_0$ and $\{C_t\}$ are taken as inputs to form $\{U_t\}$ process, and $\{U_t\}$ is used as an input into $\{R_t, V_t\}$ processes via the recursion below:

```{math}
    V_t = \left[(1-\beta)U_t^{1-\rho}+\beta R_t^{1-\rho}\right]^{\frac{1}{1-\rho}}
```

```{math}
    R_t = \mathbb{E}\left[V_{t+1}^{1-\gamma} \mid {\mathfrak F}_t\right]^{\frac{1}{1-\gamma}}
```

```{math}
    U_t = \left[(1-\alpha)C_t^{1-\epsilon}+\alpha H_t^{1-\epsilon}\right]^{\frac{1}{1-\epsilon}}
```

```{math}
    H_{t+1}  = \chi H_t + (1-\chi) C_t
```

Obviously, as $\alpha \to 0$, this preference specification degenerates to the no habit specification as described in the previous section.

We are again interested in finding a balanced growth path. In this preference specification, we can use $H$ to scale other preference variables $V$, $R$ and $U$. Since $H$ itself also grows, and $C$ is also involved here, which also grows, we can scale them by $K$, whose dynamic will be introduced soon.


```{math}
:label: ac_V_balanced
    \frac{V_t}{H_t} = \left[(1-\beta)\left(\frac{U_t}{H_t}\right)^{1-\rho}+\beta\left(\frac{R_t}{H_t}\right)^{1-\rho}\right]^{\frac{1}{1-\rho}}
```

In the special case of $\rho = 1$，

```{math}
    \frac{V_t}{H_t} = \left(\frac{U_t}{H_t}\right)^{1-\beta} \left(\frac{R_t}{H_t}\right)^{\beta}
```

```{math}
:label: ac_R_balanced
    \frac{R_t}{H_t} = \mathbb{E}\left[\left(\frac{V_{t+1}}{H_t}\right)^{1-\gamma} | {\mathfrak F}_t\right]^{\frac{1}{1-\gamma}}
```

```{math}
:label: ac_H_balanced
    \frac{H_{t+1}}{K_t}  = \chi \frac{H_t}{K_t} + (1-\chi) \frac{C_t}{K_t}
```

```{math}
:label: ac_U_balanced
    \frac{U_t}{H_t} = \left[(1-\alpha)\left(\frac{C_t}{H_t}\right)^{1-\epsilon}+\alpha\right]^{\frac{1}{1-\epsilon}}
```

In the special case of $\epsilon = 1$,

```{math}
    \frac{U_t}{H_t} = \left(\frac{C_t}{H_t}\right)^{1-\alpha}
```

(ac_model_constraint)=
## Technology: AK with adjustment cost

We consider an $AK$ model with adjustment costs and state dependent growth $G$: 

```{math}
:label: ac_feasibility
    \frac{C_t}{K_t} + \frac{I_t}{K_t}  = {\mathbf a} 
```
```{math}
:label: ac_K_growth
    \frac{K_{t+1}}{K_t}  = \left[1 + \phi_2 \left({\frac {I_t} {K_t}}\right) \right]^{\phi_1} G_{t+1}
```

Alternatively, {eq}`ac_K_growth` can be written as

```{math}
    \log{K_{t+1}} - \log{K_t} = \phi_1 \log{\left[1 + \phi_2 \left({\frac {I_t} {K_t}}\right) \right]} + \log{G_{t+1}}
```
where

```{math}
:label: ac_stochastic_growth
    G_{t+1} \equiv \exp \left( - \alpha_k + \mathbb{U}_k \cdot Z_t - {\frac 1 2} \mid \sigma_k \mid^2  + \sigma_k\cdot W_{t+1} \right)
```
and $Z_{t+1}$ is a vector exogenous process that follows

```{math}
:label: ac_exogenous
    Z_{t+1} = {\mathbb A} Z_t + \mathbb{B} W_{t+1}
```

with $Z_{1,t}$ and $Z_{2,t}$ being two components of capital growth;

$W_{t+1}$ is a shock vector containing 3 entries:

```{math}
    W_{t+1} = \left[W_{1,t+1}, W_{2,t+1}, W_{3,t+1}\right]^{\prime}
```

and they follow multivariate standard normal distribution.

```{math}
    \mathbb{A} = \begin{bmatrix}
    \exp(-\beta_1) & 0 \\ 0 & \exp(-\beta_2)
    \end{bmatrix}
```

$\mathbb{U}_k \cdot Z_t$ shifts the growth rate in technology, so it is a source of "long run risk".

When the preference shifter $\{D_t\}$ is of interest, it follows

```{math}
    \log D_{t+1} - \log D_t = \mathbb{U}_d \cdot Z_t + \sigma_d \cdot W_{t+1}
```

$\mathbb{U}_d \cdot Z_t$ shifts the growth rate in preference, so it is also a source of "long run risk".

## Stochastic Discount Factor; FOC on investment

### No household capital

The preferences described in {ref}`ac_model_preference` imply that the time $t+1$ multiplicative increment to the consumer's stochastic discount factor is (in units of $C_t$):

```{math}
    \frac{S_{t+1}}{S_t} = \beta \left(\frac{V_{t+1}}{R_t}\right)^{1-\gamma} \left(\frac{V_{t+1}}{R_t}\right)^{\rho-1} \left(\frac{C_{t+1}}{C_t}\right)^{-\rho}
```

The reason that the second term is written separately is that, it represents a change of probability measure to the shocks and thus opens the door to an expansion that we find to be revealing. The code has special treatment regarding this term.

The equilibrium $\{V_t\}$ process for a planner solves the Bellman equation

```{math}
    V_t = \max_{C_t, I_t}[(1-\beta)C_t^{1-\rho} + \beta((\mathbb{E}[V_{t+1}^{1-\gamma}| \mathfrak{F}_t])^{\frac{1}{1-\gamma}})^{1-\rho}]^{\frac{1}{1-\rho}}
```

where maximization is subject to equations in {ref}`ac_model_constraint`. The associated FOC (Euler equation) is:

```{math}
    \log\mathbb{E}\left[\frac{S_{t+1}}{S_t}\frac{MK_{t+1}}{MC_{t+1}}\psi \left(I_t, K_t, Z_t\right) \Biggl| {\mathfrak F}_t \right] = 0
```

where $MC_{t+1} = (1-\beta)C_{t+1}^{-\rho}V_{t+1}^\rho$ and $MK_{t+1} = \frac{V_{t+1}}{K_{t+1}}$ are the date $t+1$ marginal value of consumption and marginal value of capital, respectively; $\psi = - \frac{dK_{t+1}/K_t}{dC_t} = \phi_1 \phi_2 [1+\phi_2 (\frac{I_t}{K_t})^{\phi_1-1}]G_{t+1}$.

<font color='blue'>If we include preference shifter, we need to modify two terms that appear in FOC: $MC_{t+1} = (1-\beta)C_{t+1}^{-\rho}V_{t+1}^\rho D_t^{1-\rho}$ and 
```{math}
    \frac{S_{t+1}}{S_t} = \beta \left(\frac{V_{t+1}}{R_t}\right)^{1-\gamma} \left(\frac{V_{t+1}}{R_t}\right)^{\rho-1} \left(\frac{C_{t+1}}{C_t}\right)^{-\rho} \left(\frac{D_{t+1}}{D_t}\right)^{1-\rho}
```
</font>

### With household capital

Now we follow the preferences described in {ref}`ac_model_preference_capital` instead.

SDF increment in units of $U_t$:
```{math}
    \widetilde{\frac{S_{t+1}}{S_t}} = \beta \left(\frac{V_{t+1}}{R_t}\right)^{1-\gamma} \left(\frac{V_{t+1}}{R_t}\right)^{\rho-1} \left(\frac{U_{t+1}}{U_t}\right)^{-\rho}
```

But we are more interested in viewing $C_t$ as the numeraire. This leads us to introduce two additional equations in which enduring effects of consumption at $t$ come into play. These equations in effect pin down two marginal rates of substitution, $\frac{MC_t}{MU_t}$ and $\frac{MH_t}{MU_t}$. They satisfy: 


```{math}
    \frac{MC_t}{MU_t} = (1-\alpha)\left( \frac{U_t}{C_t} \right)^\epsilon + (1-\chi) \mathbb{E}\left[\widetilde{\frac{S_{t+1}}{S_t}}\frac{MH_{t+1}}{MU_{t+1}} \Biggl| {\mathfrak F}_t\right]
```
where $\frac{MH_t}{MU_t}$ satisfies:

```{math}
    \frac{MH_t}{MU_t} = \alpha\left( \frac{U_t}{H_t} \right)^\epsilon + \chi \mathbb{E}\left[\widetilde{\frac{S_{t+1}}{S_t}}\frac{MH_{t+1}}{MU_{t+1}} \Biggl| {\mathfrak F}_t \right]
```

Then we have SDF increment in units of $C_t$:

```{math}
\frac{S_{t+1}}{S_t} = \widetilde{\left(\frac{S_{t+1}}{S_t}\right)}\left(\frac{{MC_{t+1}}/{MU_{t+1}}}{{MC_t}/{MU_t}}\right)
```

The FOC again takes the same form:

```{math}
:label: ac_FOC
\log \mathbb{E}\left[\frac{S_{t+1}}{S_t}\frac{MK_{t+1}}{MC_{t+1}}\psi \left(I_t, K_t, Z_t\right) \Biggl| {\mathfrak F}_t \right] = 0
```

where $MC_{t+1} = (1-\beta)C_{t+1}^{-\rho}V_{t+1}^\rho$ and $\color{red}{MK_{t+1} = \frac{V_{t+1}}{K_{t+1}} - MH_{t+1}\frac{H_{t+1}}{K_{t+1}}}$ are the date $t+1$ marginal value of consumption and marginal value of capital, respectively; $\psi = - \frac{dK_{t+1}/K_t}{dC_t} = \phi_1 \phi_2 [1+\phi_2 (\frac{I_t}{K_t})^{\phi_1-1}]G_{t+1}$. <font color='red'>NB: $MK_{t+1}$ here is different from that in Section 1.4.1.</font>
