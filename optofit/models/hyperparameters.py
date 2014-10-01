# Create a bunch of default hyperparameters.
# These could be modified by spearmint during BO.
from parameters import Parameter

# Make a list of hypers. At the end this will be converted
# to a dict.
hl = []

### Compartment hyperparameters ###
# Capacitance of a compartment
hl.append(Parameter('C', 1.0))

### Channel hyperparameters ###
# Leak hypers
hl.append(Parameter('E_leak', -60.0))
hl.append(Parameter('a_g_leak', 2.0, lb=1.0))
hl.append(Parameter('b_g_leak', 10.0, lb=0.0))

# Na
hl.append(Parameter('E_Na', 50.0))
hl.append(Parameter('a_g_na', 5.0, lb=1.0))
hl.append(Parameter('b_g_na', 0.33, lb=0.0))

# Ca3 Na
hl.append(Parameter('E_Na', 50.0))
hl.append(Parameter('a_g_ca3na', 5.0, lb=1.0))
hl.append(Parameter('b_g_ca3na', 0.33, lb=0.0))

# Kdr
hl.append(Parameter('E_K', -77.0))
hl.append(Parameter('a_g_kdr', 6.0, lb=1.0))
hl.append(Parameter('b_g_kdr', 1.0, lb=0.0))

# Ca3 Kdr
hl.append(Parameter('E_K', -80.0))
hl.append(Parameter('a_g_ca3kdr', 6.0, lb=1.0))
hl.append(Parameter('b_g_ca3kdr', 1.0, lb=0.0))

# Ca3 Ka
hl.append(Parameter('E_K', -80.0))
hl.append(Parameter('a_g_ca3ka', 2.0, lb=1.0))
hl.append(Parameter('b_g_ca3ka', 2.0, lb=0.0))

# Ca3 Ca
hl.append(Parameter('E_Ca', 80.0))
hl.append(Parameter('a_g_ca3ca', 2.0, lb=1.0))
hl.append(Parameter('b_g_ca3ca', 2.0, lb=0.0))

# Ca3 Kahp
hl.append(Parameter('E_Kahp', -80))
hl.append(Parameter('a_g_ca3kahp', 2.0, lb=1.0))
hl.append(Parameter('b_g_ca3kahp', 2.0, lb=0.0))

# Ca3 Kc
hl.append(Parameter('E_Ca3Kc', -80))
hl.append(Parameter('a_g_ca3kc', 2.0, lb=1.0))
hl.append(Parameter('b_g_ca3kc', 2.0, lb=0.0))

# ChR2
hl.append(Parameter('E_ChR2', 0))
hl.append(Parameter('a_g_chr2', 4.0, lb=1.0))
hl.append(Parameter('b_g_chr2', 2.0, lb=0.0))

### Noisy Dynamics Parameters ###
hl.append(Parameter('N_particles', 1000, lb=10))
hl.append(Parameter('sig_V_init', 5.0, lb=0.0))
hl.append(Parameter('sig_ch_init', 0.01, lb=0.0))
hl.append(Parameter('a_sig_V', 1.0, lb=0.0))
hl.append(Parameter('b_sig_V', 1.0, lb=0.0))
hl.append(Parameter('sig_V', 5.0, lb=0.0))
hl.append(Parameter('sig_ch', 0.01, lb=0.0))

### Noisy Observation Parameters ###
hl.append(Parameter('N_particles', 1000, lb=10))
hl.append(Parameter('sig_obs_V', 5.0, lb=0.0))


### Convert this to a dict
hypers = {}
for h in hl:
    hypers[h.name] = h
