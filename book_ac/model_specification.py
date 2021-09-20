import autograd.numpy as anp
import numpy as np
from scipy import optimize

def eq_cond_no_habit(X_t, X_tp1, W_tp1, q, *args):
    # Parameters for the model
    γ, ρ, β, a, ϕ_1, ϕ_2, α_k, U_k, σ_k, A, B = args

    # Variables:
    # log V_t/C_t, log R_t/C_t,
    # log C_t/K_t, log I_t/K_t,
    # log (K_{t}/K_{t-1}), Z_{1,t}, Z_{2,t}
    vmc_t, rmc_t, cmk_t, imk_t, gk_t, z1_t, z2_t = X_t.ravel()
    vmc_tp1, rmc_tp1, cmk_tp1, imk_tp1, gk_tp1, z1_tp1, z2_tp1 = X_tp1.ravel()

    # Exogenous states
    Z_t = anp.array([z1_t, z2_t])
    Z_tp1 = anp.array([z1_tp1, z2_tp1])
    # Stochastic depreciation, capital growth and log ψ
    g_dep = -α_k + U_k.T@Z_t + σ_k.T@W_tp1
    log_ψ = anp.log(ϕ_1*ϕ_2) + (ϕ_1-1)*anp.log(1+ϕ_2*anp.exp(imk_t)) + g_dep
    # log SDF, excluding the change of measure
    sdf_ex = anp.log(β) + (ρ-1)*(vmc_tp1+cmk_tp1+gk_tp1-cmk_t-rmc_t) - ρ*(cmk_tp1+gk_tp1-cmk_t)

    # Marginals and pricing kernel
    mk_tp1 = vmc_tp1+cmk_tp1
    mc_tp1 = anp.log(1-β) + ρ*(vmc_tp1)
    log_Q = sdf_ex + mk_tp1 - mc_tp1

    # Eq0: Change of measure evaluated at γ=0
    m = vmc_tp1 + cmk_tp1 + gk_tp1 - cmk_t - rmc_t
    # Eq1: Recursive utility
    res_1 = (1-β) + β*anp.exp((1-ρ)*(rmc_t)) - anp.exp((1-ρ)*(vmc_t))
    # Eq2: FOC for consumption/investment
    res_2 = anp.exp(log_Q + log_ψ)
    # Eq3: Investment ratio
    res_3 = a - anp.exp(cmk_t) - anp.exp(imk_t)
    # Eq4: capital
    res_4 = gk_tp1 - ϕ_1 * anp.log(1+ϕ_2*anp.exp(imk_t)) - g_dep
    # Eq5-6: State process
    res_5 = (A@Z_t + B@W_tp1 - Z_tp1)[0]
    res_6 = (A@Z_t + B@W_tp1 - Z_tp1)[1]

    return anp.array([m, res_1,res_2,res_3,res_4,res_5, res_6])

def ss_func_no_habit(*args):
    # Extra parameters for the model
    γ, ρ, β, a, ϕ_1, ϕ_2, α_k, U_k, σ_k, A, B = args

    # Optimize over c_t-k_t
    def f(cmk):
        # Level investment
        I = a - np.exp(cmk)
        # Capital growth
        g_k = ϕ_1 * np.log(1 + ϕ_2 * I) - α_k
        # Set growth rate to capital growth
        η = g_k
        # Increment in capital induced by a marginal decrease in consumption
        log_ψ =  np.log(ϕ_1) + np.log(ϕ_2) + (ϕ_1-1)*np.log(1 + ϕ_2 * I) - α_k
        # v
        vmc = (np.log(1-β) - np.log(1-β*np.exp((1-ρ)*η)))/(1-ρ)
        # sdf, note that sdf_c = sdf_u in steady states
        sdf = np.log(β) - ρ*η
        # log_Q
        mk_next = vmc+cmk
        mc_next = np.log(1-β) + ρ*vmc
        log_Q = mk_next - mc_next + sdf
        return np.exp(log_Q + log_ψ) - 1

    # Find roots
    cmk_star = optimize.bisect(f,-40,np.log(a))
    cmk = cmk_star

    # Calculate steady states
    z_1 = 0.
    z_2 = 0.
    Z = np.array([z_1,z_2])
    I = a - np.exp(cmk)
    g_k = ϕ_1 * np.log(1 + ϕ_2 * I) - α_k
    η = g_k
    
    # c, k, h, u, sdf, v, r, mu, mh, mc, imk
    vmc = (np.log(1-β) - np.log(1-β*np.exp((1-ρ)*η)))/(1-ρ)
    rmc = vmc + η
    imk = np.log(a - np.exp(cmk))

    X_0 = np.array([vmc,rmc,cmk,imk,g_k,z_1,z_2])
    return X_0


def eq_cond_internal_habit(X_t, X_tp1, W_tp1, q, *args):
    # Parameters for the model
    γ, ρ, β, χ, α, ϵ, a, ϕ_1, ϕ_2, α_k, U_k, σ_k, A, B = args

    # Variables:
    # log V_t/H_t, log R_t/H_t, log U_t/H_t, log C_t/K_t,
    # log I_t/K_t, log MH_t/MU_t, log MC_t/MU_t,
    # log H_t/K_t, log (K_{t}/K_{t-1}), Z_{1,t}, Z_{2,t}
    vmh_t, rmh_t, umh_t, cmk_t, imk_t, mhmu_t,\
        mcmu_t, hmk_t, gk_t, z1_t, z2_t = X_t.ravel()
    vmh_tp1, rmh_tp1, umh_tp1, cmk_tp1, imk_tp1,\
        mhmu_tp1, mcmu_tp1, hmk_tp1, gk_tp1, z1_tp1, z2_tp1 = X_tp1.ravel()

    # Exogenous states
    Z_t = anp.array([z1_t, z2_t])
    Z_tp1 = anp.array([z1_tp1, z2_tp1])
    # Stochastic depreciation, capital growth and log ψ
    g_dep = -α_k + U_k.T@Z_t + σ_k.T@W_tp1
    log_ψ = anp.log(ϕ_1*ϕ_2) + (ϕ_1-1)*anp.log(1+ϕ_2*anp.exp(imk_t)) + g_dep
    # log SDF in units of U, excluding the change of measure
    sdf_u = anp.log(β) + (ρ-1)*(vmh_tp1+hmk_tp1+gk_tp1-hmk_t-rmh_t) - ρ*(umh_tp1+hmk_tp1+gk_tp1-hmk_t-umh_t)
    sdf_ex = sdf_u + mcmu_tp1 - mcmu_t

    # Marginals and pricing kernel
    mu_tp1 = anp.log(1-β) + ρ*(vmh_tp1-umh_tp1)
    mk_tp1 = anp.log(anp.exp(vmh_tp1+hmk_tp1)-anp.exp(mhmu_tp1+mu_tp1+hmk_tp1))
    log_Q = sdf_ex + mk_tp1 - (mcmu_tp1+mu_tp1)

    # Eq0: Change of measure evaluated at γ=0
    m = vmh_tp1 + hmk_tp1 + gk_tp1 - hmk_t - rmh_t
    # Eq1: Recursive utility
    res_1 = (1-β)*anp.exp((1-ρ)*(umh_t)) + β*anp.exp((1-ρ)*(rmh_t)) - anp.exp((1-ρ)*(vmh_t))
    # Eq2: Utility function
    res_2 = anp.log((1-α)*anp.exp((1-ϵ)*(cmk_t-hmk_t))+α)/(1-ϵ) - (umh_t)
    # Eq3: FOC for consumption/investment
    res_3 = anp.exp(log_Q + log_ψ)
    # Eq4: MC/MU
    res_4 = (1-α)*anp.exp(ϵ*(umh_t+hmk_t-cmk_t)) + (1-χ)*anp.exp(sdf_u+mhmu_tp1) - anp.exp(mcmu_t)
    # Eq5: MH/MU
    res_5 = α*anp.exp(ϵ*(umh_t)) + χ*anp.exp(sdf_u+mhmu_tp1) - anp.exp(mhmu_t)
    # Eq6: Investment ratio
    res_6 = a - anp.exp(cmk_t) - anp.exp(imk_t)
    # Eq7: Habit evolution
    res_7 = anp.exp(hmk_tp1+gk_tp1) - χ*anp.exp(hmk_t) - (1-χ)*anp.exp(cmk_t)
    # Eq8: capital
    res_8 = gk_tp1 - ϕ_1 * anp.log(1+ϕ_2*anp.exp(imk_t)) - g_dep
    # Eq9-10: State process
    res_9 = (A@Z_t + B@W_tp1 - Z_tp1)[0]
    res_10 = (A@Z_t + B@W_tp1 - Z_tp1)[1]

    return anp.array([m, res_1,res_2,res_3,res_4,res_5,res_6,res_7,res_8,res_9, res_10])

def ss_func_internal_habit(*args):
    # Extra parameters for the model
    γ, ρ, β, χ, α, ϵ, a, ϕ_1, ϕ_2, α_k, U_k, σ_k, A, B = args

    # Optimize over c_t-k_t
    def f(cmk):
        # Level investment
        I = a - np.exp(cmk)
        # Capital growth
        g_k = ϕ_1 * np.log(1 + ϕ_2 * I) - α_k
        # Set growth rate to capital growth
        η = g_k
        # Increment in capital induced by a marginal decrease in consumption
        log_ψ =  np.log(ϕ_1) + np.log(ϕ_2) + (ϕ_1-1)*np.log(1 + ϕ_2 * I) - α_k
        # c, h, u, v
        c = 0.
        hmk = np.log((1-χ)*np.exp(cmk)/(np.exp(η)-χ))
        umh = np.log((1-α)*np.exp((1-ϵ)*(cmk-hmk))+α)/(1-ϵ)
        vmh = (np.log(1-β) +(1-ρ)*umh- np.log(1-β*np.exp((1-ρ)*η)))/(1-ρ)
        # sdf, note that sdf_c = sdf_u in steady states
        sdf = np.log(β) - ρ*η
        # mu, mh, mc
        MHMU = α*np.exp(ϵ*(umh))/(1-χ*np.exp(sdf))
        mhmu = np.log(MHMU)
        MCMU = (1-α)*np.exp(ϵ*(umh+hmk-cmk)) + (1-χ)*np.exp(sdf)*MHMU
        mcmu = np.log(MCMU)
        # log_Q
        mu = np.log(1-β) + ρ*(vmh-umh)
        mk_next = np.log(np.exp(vmh+hmk)-np.exp(mhmu+mu+hmk))
        mc_next = mcmu + mu
        log_Q = mk_next - mc_next + sdf
        return np.exp(log_Q + log_ψ) - 1

    # Find roots
    cmk_star = optimize.bisect(f,-40,np.log(a))
    cmk = cmk_star

    # Calculate steady states
    z_1 = 0.
    z_2 = 0.
    Z = np.array([z_1,z_2])
    I = a - np.exp(cmk)
    g_k = ϕ_1 * np.log(1 + ϕ_2 * I) - α_k
    η = g_k
    
    hmk = np.log((1-χ)*np.exp(cmk)/(np.exp(η)-χ))
    umh = np.log((1-α)*np.exp((1-ϵ)*(cmk-hmk))+α)/(1-ϵ)
    δ = -np.log(β)
    sdf = -δ - ρ*η
    vmh = (np.log(1-β) +(1-ρ)*umh- np.log(1-β*np.exp((1-ρ)*η)))/(1-ρ)
    rmh = vmh + η
    MHMU = α*np.exp(ϵ*(umh))/(1-χ*np.exp(sdf))
    mhmu = np.log(MHMU)
    MCMU = (1-α)*np.exp(ϵ*(umh+hmk-cmk)) + (1-χ)*np.exp(sdf)*MHMU
    mcmu = np.log(MCMU)
    imk = np.log(a - np.exp(cmk))

    X_0 = np.array([vmh,rmh,umh,cmk,imk,mhmu,mcmu,hmk,g_k,z_1,z_2])
    return X_0



def log_SDF_ex_no_habit(X_t, X_tp1, W_tp1, q, *args):
    γ, ρ, β, a, ϕ_1, ϕ_2, α_k, U_k, σ_k, A, B = args
    
    vmc_t, rmc_t, cmk_t, imk_t, gk_t, z1_t, z2_t = X_t.ravel()
    vmc_tp1, rmc_tp1, cmk_tp1, imk_tp1, gk_tp1, z1_tp1, z2_tp1 = X_tp1.ravel()

    sdf_ex = anp.log(β) + (ρ-1)*(vmc_tp1+cmk_tp1+gk_tp1-cmk_t-rmc_t) - ρ*(cmk_tp1+gk_tp1-cmk_t)
    
    return sdf_ex

def log_SDF_ex_internal_habit(X_t, X_tp1, W_tp1, q, *args):
    γ, ρ, β, χ, α, ϵ, a, ϕ_1, ϕ_2, α_k, U_k, σ_k, A, B = args
    
    vmh_t, rmh_t, umh_t, cmk_t, imk_t, mhmu_t,\
        mcmu_t, hmk_t, gk_t, z1_t, z2_t = X_t.ravel()
    vmh_tp1, rmh_tp1, umh_tp1, cmk_tp1, imk_tp1,\
        mhmu_tp1, mcmu_tp1, hmk_tp1, gk_tp1, z1_tp1, z2_tp1 = X_tp1.ravel()

    sdf_u = anp.log(β) + (ρ-1)*(vmh_tp1+hmk_tp1+gk_tp1-hmk_t-rmh_t) - ρ*(umh_tp1+hmk_tp1+gk_tp1-hmk_t-umh_t)
    sdf_ex = sdf_u + mcmu_tp1 - mcmu_t
    
    return sdf_ex


def eq_cond_no_habit_with_pref_shock(X_t, X_tp1, W_tp1, q, *args):
    # Parameters for the model
    γ, ρ, β, a, ϕ_1, ϕ_2, α_k, U_k, σ_k, U_d, σ_d, A, B = args

    # Variables in X_t:
    # log V_t/C_t, log R_t/C_t,
    # log C_t/K_t, log I_t/K_t,
    # log (K_{t}/K_{t-1}), Z_{1,t}, Z_{2,t}
    vmcd_t, rmcd_t, cmk_t, imk_t, gk_t, z1_t, z2_t = X_t.ravel()
    vmcd_tp1, rmcd_tp1, cmk_tp1, imk_tp1, gk_tp1, z1_tp1, z2_tp1 = X_tp1.ravel()

    # Exogenous states
    Z_t = anp.array([z1_t, z2_t])
    Z_tp1 = anp.array([z1_tp1, z2_tp1])
    gd_tp1 = U_d.T@Z_t + σ_d.T@W_tp1
    # Stochastic depreciation, capital growth and log ψ
    g_dep = -α_k + U_k.T@Z_t + σ_k.T@W_tp1
    log_ψ = anp.log(ϕ_1*ϕ_2) + (ϕ_1-1)*anp.log(1+ϕ_2*anp.exp(imk_t)) + g_dep
    # log SDF, excluding the change of measure
    sdf_ex = anp.log(β) + (ρ-1)*(vmcd_tp1+cmk_tp1+gk_tp1 + gd_tp1 -cmk_t-rmcd_t) \
            - ρ*(cmk_tp1+gk_tp1-cmk_t) + (1-ρ)*gd_tp1

    # Marginals and pricing kernel
#     mk_tp1 = vmc_tp1+cmk_tp1
#     mc_tp1 = anp.log(1-β) + ρ*(vmc_tp1)  + (1-ρ)*(dcmk_tp1 - cmk_tp1)
#     log_Q = sdf_ex + mk_tp1 - mc_tp1
    mk_mc_tp1 = -anp.log(1-β) + (1-ρ)*vmcd_tp1 + cmk_tp1
    log_Q = sdf_ex + mk_mc_tp1

    # Eq0: Change of measure evaluated at γ=0
    m = vmcd_tp1 + cmk_tp1 + gk_tp1 + gd_tp1 - cmk_t - rmcd_t
    # Eq1: Recursive utility
    res_1 = (1-β) + β*anp.exp((1-ρ)*(rmcd_t)) - anp.exp((1-ρ)*(vmcd_t))
    # Eq2: FOC for consumption/investment
    res_2 = anp.exp(log_Q + log_ψ)
    # Eq3: Investment ratio
    res_3 = a - anp.exp(cmk_t) - anp.exp(imk_t)
    # Eq4: capital
    res_4 = gk_tp1 - ϕ_1 * anp.log(1+ϕ_2*anp.exp(imk_t)) - g_dep
    # Eq5-6: State process
    res_5 = (A@Z_t + B@W_tp1 - Z_tp1)[0]
    res_6 = (A@Z_t + B@W_tp1 - Z_tp1)[1]
    
#     res_7 = dcmk_tp1 - cmk_tp1 - (dcmk_t - cmk_t) - U_d.T@Z_t - σ_d.T@W_tp1

    return anp.array([m, res_1,res_2,res_3,res_4,res_5, res_6])
    
    
def ss_func_no_habit_with_pref_shock(*args):
    # Extra parameters for the model
    γ, ρ, β, a, ϕ_1, ϕ_2, α_k, U_k, σ_k, U_d, σ_d, A, B = args

    # Optimize over c_t-k_t
    def f(cmk):
        # Level investment
        I = a - np.exp(cmk)
        # Capital growth
        g_k = ϕ_1 * np.log(1 + ϕ_2 * I) - α_k
        # Set growth rate to capital growth
        η = g_k
        # Increment in capital induced by a marginal decrease in consumption
        log_ψ =  np.log(ϕ_1) + np.log(ϕ_2) + (ϕ_1-1)*np.log(1 + ϕ_2 * I) - α_k
        # v
        vmc = (np.log(1-β) - np.log(1-β*np.exp((1-ρ)*η)))/(1-ρ)
        # sdf, note that sdf_c = sdf_u in steady states
        sdf = np.log(β) - ρ*η
        # log_Q
        mk_next = vmc+cmk
        mc_next = np.log(1-β) + ρ*vmc
        log_Q = mk_next - mc_next + sdf
        return np.exp(log_Q + log_ψ) - 1

    # Find roots
    cmk_star = optimize.bisect(f,-40,np.log(a))
    cmk = cmk_star

    # Calculate steady states
    z_1 = 0.
    z_2 = 0.
    Z = np.array([z_1,z_2])
    I = a - np.exp(cmk)
    g_k = ϕ_1 * np.log(1 + ϕ_2 * I) - α_k
    η = g_k
    
    # c, k, h, u, sdf, v, r, mu, mh, mc, imk
    vmcd = (np.log(1-β) - np.log(1-β*np.exp((1-ρ)*η)))/(1-ρ)
    rmcd = vmcd + η
    imk = np.log(a - np.exp(cmk))

    X_0 = np.array([vmcd,rmcd,cmk,imk, g_k,z_1,z_2])
    return X_0
    
    
def log_SDF_ex_no_habit_with_pref_shock(X_t, X_tp1, W_tp1, q, *args):
    γ, ρ, β, a, ϕ_1, ϕ_2, α_k, U_k, σ_k, U_d, σ_d, A, B = args
    
    vmcd_t, rmcd_t, cmk_t, imk_t, gk_t, z1_t, z2_t = X_t.ravel()
    vmcd_tp1, rmcd_tp1, cmk_tp1, imk_tp1, gk_tp1, z1_tp1, z2_tp1 = X_tp1.ravel()

    Z_t = anp.array([z1_t, z2_t])
    gd_tp1 = U_d.T@Z_t + σ_d.T@W_tp1
    sdf_ex = anp.log(β) + (ρ-1)*(vmcd_tp1+cmk_tp1+gk_tp1 + gd_tp1 -cmk_t-rmcd_t) \
        - ρ*(cmk_tp1+gk_tp1-cmk_t) + (1-ρ)*gd_tp1
    
    return sdf_ex
