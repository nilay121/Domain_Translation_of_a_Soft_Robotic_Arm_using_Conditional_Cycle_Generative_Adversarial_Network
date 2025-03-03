import sys
sys.path.append('../')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch 
from torch.utils.data import DataLoader

from utils.utils import gradient_penalty, plotGAN_loss, TaskSpacePlotterAllpred, \
                        TaskSpacePlotter, TSTime, ErrorTime, plot3dTrajectory, \
                        PressurePlotterCS, processedData, addFromInterpolatedData,\
                        sampleFromInterpolatedData, orientationError, interpolate_actuation,\
                        save_dictionary, load_dictionary
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter

def closedLoopModelPred(fm2d2_input, device):
    fm2d2 = torch.load("../forwardModels/fmd2.pickle").to(device)
    fm2d2.eval()
    with torch.no_grad():
        # forward model predictions
        pred_fm2d2 = fm2d2(fm2d2_input)
    fm2d2.train()
    return pred_fm2d2

def reset_isupport_model(Isupport_sim, numeric_parameters, geometric_parameters, material_parametersD2, 
                                valve_parameters, environment_parameters, simulation_parametersD2):
    isupportFM2D2 = Isupport_sim(numeric_parameters, geometric_parameters, material_parametersD2, 
                                valve_parameters, environment_parameters, simulation_parametersD2)
    isupportFM2D2.finalize()
    return isupportFM2D2

def deterministic_robotModel(Isupport_sim, teq, predictedPressure):
    time_delay_list = teq * np.ones(predictedPressure.shape[0])
    _, _ = Isupport_sim.inflate_sequence(commands=predictedPressure, 
                                            command_delays=time_delay_list)
    ## First module
    rod_history1 = Isupport_sim.rod_history1 
    ee_pos = np.array(rod_history1['position'])[:-1, [2,0,1], -1][-predictedPressure.shape[0]:]
    # tip positions of first module --> slice to get the current batch trajectory
    ee_d3 = np.array(rod_history1["director"])[:-1, 2, [2, 0, 1], -1][-predictedPressure.shape[0]:] 
    pred_combinedPosD2 = np.concatenate([ee_pos, ee_d3], axis=1)
    return pred_combinedPosD2

def interpolation_steps_calc(lower_interpolate_step, upper_interpolate_step, Isupport_sim, seq_length, 
                            numeric_parameters, geometric_parameters, material_parametersD2, 
                            valve_parameters, environment_parameters, simulation_parametersD2,
                            model_gen2, teqD2, mse_loss, interpolation_info, device):
    step_size = 5 #3

    # Load the test shape and forward model
    traj_collector_d1 = np.array(pd.read_csv("../../../../../../Babbling_data/domain1/rect30s_test_1Mod_TS.csv"))
    pressure_collector_d1 = np.array(pd.read_csv("../../../../../../Babbling_data/domain1/rect30s_test_1Mod_AS.csv"))

    #convert to torch
    traj_collector_d1 = torch.tensor(traj_collector_d1.astype("float32"))
    pressure_collector_d1 = torch.tensor(pressure_collector_d1.astype("float32"))
    current_batch = traj_collector_d1.shape[0]

    # Loop through all the possible interpolation steps
    steps = np.arange(lower_interpolate_step, upper_interpolate_step, step_size)
    for i in steps:
        # print("Current interpolation step ", i)
        gen2_input_data = torch.cat([pressure_collector_d1, traj_collector_d1], axis=1).reshape(current_batch, 
                                        seq_length, -1).to(device) 
        # Gen2 model predictions
        model_gen2.eval()
        with torch.no_grad():
            gen2_pred_pressure = model_gen2(gen2_input_data, True, i).detach().cpu().numpy()
            # gen2_pred_pressure = self.model_gen2(gen2_input_data, True, i)
        model_gen2.train()

        # # # Forward model --> LSTM based model
        # prev_XeeD2 = torch.tile(torch.tensor([0,0,-0.38,0,0,-1]), (gen2_pred_pressure.shape[0], 1)).to(self.device)
        # fm2d2_input = torch.concat([prev_XeeD2, gen2_pred_pressure[:, 0:3]/3.5], 
        #                             axis=1).reshape(gen2_pred_pressure.shape[0], self.seq_length, -1)
        # pred_combinedPosFM2D2 = self.closedLoopModelPred(fm2d2_input)
        
        # Forward model --> Deterministic model
        isupportFM2D2 = reset_isupport_model(Isupport_sim, numeric_parameters, geometric_parameters, material_parametersD2, 
                            valve_parameters, environment_parameters, simulation_parametersD2)
        pred_combinedPosFM2D2 = deterministic_robotModel(Isupport_sim=isupportFM2D2, 
                                                    teq=teqD2, 
                                                    predictedPressure=gen2_pred_pressure)
        # Sample from the data
        pred_combinedPosFM2D2 = pred_combinedPosFM2D2
        _, pred_fm2d2 = sampleFromInterpolatedData(pred_combinedPosFM2D2, None, i)
        # pred_fm2d2 = addFromInterpolatedData(pred_combinedPosFM2D2, i)
        pred_fm2d2 = torch.tensor(pred_fm2d2)

        # Error
        fm2d2_l2_error = mse_loss(pred_fm2d2, traj_collector_d1)
        # fm2d2_l2_error = np.linalg.norm((traj_collector_d1[:, 0:3].detach().cpu().numpy()-pred_fm2d2[:, 0:3]), 
        #                                 axis=1).mean() 
        # Add to dictionary
        interpolation_info[i] = fm2d2_l2_error
    interpolate_steps = list(interpolation_info.keys())[np.argmin(list(interpolation_info.values()))]
    print("The final interpolation steps is ", interpolate_steps)

    # Reset the robot to rest position
    reset_isupport_model(Isupport_sim, numeric_parameters, geometric_parameters, material_parametersD2, 
                            valve_parameters, environment_parameters, simulation_parametersD2)

    return interpolate_steps


def metricsReport(pressureD1, GANpredPressure_d2, Xee_d1, pred_combinedPosFM2D2, 
                    pred_combinedPosFM2D1, pred_combinedPosFM1D1, shapeName, save_errors):
    ## D2 Pressure
    rmse_PD2 = mean_squared_error(pressureD1, GANpredPressure_d2, squared=False).mean()  

    ## D1 D2 EE positions error
    rmse_Xeefm2d2 = np.linalg.norm((Xee_d1[:, 0:3]-pred_combinedPosFM2D2[:, 0:3]), axis=1).mean() 
    rmse_Xeefm2d1 = np.linalg.norm((Xee_d1[:, 0:3]-pred_combinedPosFM2D1[:, 0:3]), axis=1).mean()
    random_rmseXee = np.linalg.norm((Xee_d1[:, 0:3]-np.random.uniform(-1,1, (Xee_d1[:,0:3].shape[0], 
                                    Xee_d1[:,0:3].shape[1]))), axis=1).mean() 
    ## D1 D2 EE orientations error
    fm2d2_orienError = orientationError(Xee_d1, pred_combinedPosFM2D2).mean() 
    fm2d1_orienError = orientationError(Xee_d1, pred_combinedPosFM2D1).mean() 
    
    ## Save errors to file
    save_errors(rmse_Xeefm2d2=rmse_Xeefm2d2, fm2d2_orienError=fm2d2_orienError, shapeName=shapeName)

    print("Domanin2 error in pressure ", rmse_PD2)
    print("FM2D2 error in task space ", rmse_Xeefm2d2)
    print("FM2D1 error in task space ", rmse_Xeefm2d1)
    print("Random domain2 error in task space ", random_rmseXee) 
    print("Orien error for FM2D2 is ", fm2d2_orienError)
    print("Orien error for FM2D1 is ", fm2d1_orienError)

def save_logs(originalXeeD1, MLP_predXeeD2, GAN_predXeeD2, XeeD2_WithD1, shape):
    datasets = [originalXeeD1, MLP_predXeeD2, GAN_predXeeD2, XeeD2_WithD1]
    names = ["orgXeeD1", "FM2D2_bench", "FM2D2_gan", "FM2D1_orgModel"]
    for i in range(len(datasets)):
        if datasets[i] is not None:
            temp_df = pd.DataFrame(datasets[i])
            temp_df.to_csv(f"../resultFiles/{names[i]}_{shape}.csv", index=False)    
        else:
            print(f"Warning!! No data to save for {names[i]}") 

def save_errors(rmse_Xeefm2d2, fm2d2_orienError, shapeName):
    path = f"../resultFiles/error_{shapeName}.txt"
    toWrite  = f"FM2D2 Position Error {rmse_Xeefm2d2} FM2D2 orientation Error {fm2d2_orienError}"
    if os.path.isfile(path):
        print("File exists!! ")
    else:
        f = open(path, "x")
        f.write(toWrite)
        f.close()
        print("Errors saved to file !! ")