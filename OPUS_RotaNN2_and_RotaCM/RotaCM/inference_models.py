# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

from RotaCM.my_model import Model

#============================Parameters====================================
params = {}
params["filter_num"] = 64
params["num_layers"] = 61
params["dropout"] = 0.5
#============================Models====================================

model_rotacm1 = Model(params=params, name="rota_cm")
model_rotacm1.params["save_path"] = "./RotaCM/models/1"
model_rotacm1.load_model()  

model_rotacm2 = Model(params=params, name="rota_cm")
model_rotacm2.params["save_path"] = "./RotaCM/models/2"
model_rotacm2.load_model()  

model_rotacm3 = Model(params=params, name="rota_cm")
model_rotacm3.params["save_path"] = "./RotaCM/models/3"
model_rotacm3.load_model()  

model_rotacm4 = Model(params=params, name="rota_cm")
model_rotacm4.params["save_path"] = "./RotaCM/models/4"
model_rotacm4.load_model()  

model_rotacm5 = Model(params=params, name="rota_cm")
model_rotacm5.params["save_path"] = "./RotaCM/models/5"
model_rotacm5.load_model()  

model_rotacm6 = Model(params=params, name="rota_cm")
model_rotacm6.params["save_path"] = "./RotaCM/models/6"
model_rotacm6.load_model() 

model_rotacm7 = Model(params=params, name="rota_cm")
model_rotacm7.params["save_path"] = "./RotaCM/models/7"
model_rotacm7.load_model() 

def test_infer_step(x1d, x2d):
    
    rota_predictions = []
    
    rota_prediction = model_rotacm1.inference(x1d, x2d, y=None, y_mask=None, training=False)       
    rota_predictions.append(rota_prediction)

    rota_prediction = model_rotacm2.inference(x1d, x2d, y=None, y_mask=None, training=False)       
    rota_predictions.append(rota_prediction)
    
    rota_prediction = model_rotacm3.inference(x1d, x2d, y=None, y_mask=None, training=False)       
    rota_predictions.append(rota_prediction)
    
    rota_prediction = model_rotacm4.inference(x1d, x2d, y=None, y_mask=None, training=False)       
    rota_predictions.append(rota_prediction)
    
    rota_prediction = model_rotacm5.inference(x1d, x2d, y=None, y_mask=None, training=False)       
    rota_predictions.append(rota_prediction)    

    rota_prediction = model_rotacm6.inference(x1d, x2d, y=None, y_mask=None, training=False)       
    rota_predictions.append(rota_prediction) 
    
    rota_prediction = model_rotacm7.inference(x1d, x2d, y=None, y_mask=None, training=False)       
    rota_predictions.append(rota_prediction) 
    
    return rota_predictions