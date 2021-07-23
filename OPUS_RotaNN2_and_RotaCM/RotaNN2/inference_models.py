# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

from RotaNN2.my_model import Model

#============================Parameters====================================
params = {}
params["d_input"] = 41 + 128 + 512
params["1d_input"] = 41
params["d_rota_output"] = 8
params["dropout_rate"] = 0.25
  
#parameters of transfomer model
params["transfomer_layers"] = 2
params["transfomer_num_heads"] = 1

#parameters of birnn model
params["lstm_layers"] = 4
params["lstm_units"] = 1024

#============================Models====================================

model_rota1 = Model(params=params, name="rota")
model_rota1.params["save_path"] = "./RotaNN2/models/1"
model_rota1.load_model()  

model_rota2 = Model(params=params, name="rota")
model_rota2.params["save_path"] = "./RotaNN2/models/2"
model_rota2.load_model()  

model_rota3 = Model(params=params, name="rota")
model_rota3.params["save_path"] = "./RotaNN2/models/3"
model_rota3.load_model()  

model_rota4 = Model(params=params, name="rota")
model_rota4.params["save_path"] = "./RotaNN2/models/4"
model_rota4.load_model()  

model_rota5 = Model(params=params, name="rota")
model_rota5.params["save_path"] = "./RotaNN2/models/5"
model_rota5.load_model()  

model_rota6 = Model(params=params, name="rota")
model_rota6.params["save_path"] = "./RotaNN2/models/6"
model_rota6.load_model()  

model_rota7 = Model(params=params, name="rota")
model_rota7.params["save_path"] = "./RotaNN2/models/7"
model_rota7.load_model()  

def test_infer_step(x, x_mask, x_trr):
    
    rota_predictions = []
    
    rota_prediction = model_rota1.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    rota_predictions.append(rota_prediction)
    
    rota_prediction = model_rota2.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    rota_predictions.append(rota_prediction)

    rota_prediction = model_rota3.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    rota_predictions.append(rota_prediction)

    rota_prediction = model_rota4.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    rota_predictions.append(rota_prediction)
    
    rota_prediction = model_rota5.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    rota_predictions.append(rota_prediction)
    
    rota_prediction = model_rota6.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    rota_predictions.append(rota_prediction)

    rota_prediction = model_rota7.inference(x, x_mask, x_trr, y=None, y_mask=None, training=False)        
    rota_predictions.append(rota_prediction)
    
    return rota_predictions