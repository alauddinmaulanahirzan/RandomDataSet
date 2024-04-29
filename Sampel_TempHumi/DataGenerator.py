import pandas as pd
import numpy as np
import lzma
import pickle
import numpy as np
import skfuzzy as fuzz
from os.path import exists
from skfuzzy import control as ctrl

# Antecedents (inputs)
temp = ctrl.Antecedent(np.arange(0, 101, 1), 'temp')
humi = ctrl.Antecedent(np.arange(0, 101, 1), 'humi')

# Consequent (output)
comfort = ctrl.Consequent(np.arange(0, 5, 1), 'comfort')

# Membership functions
temp['cold'] = fuzz.trapmf(temp.universe, [0, 0, 10, 15])
temp['warm'] = fuzz.trapmf(temp.universe, [10, 15, 30, 35])
temp['hot'] = fuzz.trapmf(temp.universe, [30, 35, 100, 100])

humi['low'] = fuzz.trapmf(humi.universe, [0, 0, 30, 35])
humi['medium'] = fuzz.trapmf(humi.universe, [30, 35, 60, 65])
humi['high'] = fuzz.trapmf(humi.universe, [60, 65, 100, 100])

comfort['low'] = fuzz.trimf(comfort.universe, [0, 1, 2])
comfort['medium'] = fuzz.trimf(comfort.universe, [1, 2, 3])
comfort['high'] = fuzz.trimf(comfort.universe, [2, 3, 4])

# Combination based rules
rule1 = ctrl.Rule(temp['cold'] & humi['low'], comfort['low'])
rule2 = ctrl.Rule(temp['cold'] & humi['medium'], comfort['low'])
rule3 = ctrl.Rule(temp['cold'] & humi['high'], comfort['medium'])

rule4 = ctrl.Rule(temp['warm'] & humi['low'], comfort['high'])
rule5 = ctrl.Rule(temp['warm'] & humi['medium'], comfort['high'])
rule6 = ctrl.Rule(temp['warm'] & humi['high'], comfort['medium'])

rule7 = ctrl.Rule(temp['hot'] & humi['low'], comfort['low'])
rule8 = ctrl.Rule(temp['hot'] & humi['medium'], comfort['low'])
rule9 = ctrl.Rule(temp['hot'] & humi['high'], comfort['low'])

crt_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,
                               rule4, rule5, rule6,
                               rule7, rule8, rule9])

crt_ctrl_simulation = ctrl.ControlSystemSimulation(crt_ctrl)

# Data Generator

data = []
index = 1

for temp_val in np.arange(0, 101, 1):
    for humi_val in np.arange(0, 101, 1):
        # Get Fuzzy Value
        crt_ctrl_simulation.input['temp'] = temp_val
        crt_ctrl_simulation.input['humi'] = humi_val
        crt_ctrl_simulation.compute()
        # Output
        output = crt_ctrl_simulation.output['comfort']

        data.append([index, temp_val, humi_val, round(output)])
        index += 1

dataset = pd.DataFrame(data, columns=['No', "Suhu", "Lembab", "Nyaman"])
dataset.set_index("No", inplace=True)
dataset.to_csv("Sampel.csv", sep=';', quotechar='"')
