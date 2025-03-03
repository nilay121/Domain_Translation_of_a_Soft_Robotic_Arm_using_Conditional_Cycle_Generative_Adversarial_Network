import sys
sys.path.append('../')

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import logging
from pathlib import Path
from tqdm import tqdm
import torch 
import torch.nn as nn
from torch.nn import BCELoss
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from torch.optim import Adam,SGD, AdamW
from sklearn.preprocessing import MinMaxScaler
from utils.i_supportModelParam import ISupportParamD1, ISupportParamD2
from utils.models import MLPd1T1, Generator, Discriminator, rnn_forwardModelD1, \
                        Reshape
from utils.utils import gradient_penalty, plotGAN_loss, TaskSpacePlotterAllpred, \
                        TaskSpacePlotter, TSTime, ErrorTime, plot3dTrajectory, \
                        PressurePlotterCS, processedData, addFromInterpolatedData,\
                        sampleFromInterpolatedData, orientationError, interpolate_actuation,\
                        save_dictionary, load_dictionary
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.arm_1 import ISupportElastica
from train_test import train_model, customShape_test
from torch.utils.tensorboard import SummaryWriter


## Disable Elastica warnings
logger = logging.getLogger()
logger.disabled = True

class JointTraining:
    def __init__(self, modelMLPd1, GAN_model1, GAN_model2, DISC_model1, DISC_model2, scaler, epochs, critic_epochs, 
                 learning_rate, lr_generator, lr_disc, batch_size, testBatchSize, patience, numeric_parameter, geometric_parameter, 
                 material_parametersD1, material_parametersD2, valve_parameters, environment_parameters, joint_parameters, 
                 simulation_parametersD1, simulation_parametersD2, pressure_input_dim, model_interpolation, 
                 device):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_generator = lr_generator
        self.lr_disc = lr_disc
        self.batch_size = batch_size
        self.patience = patience
        self.numeric_parameters = numeric_parameter
        self.geometric_parameters = geometric_parameter
        self.material_parametersD1 = material_parametersD1
        self.material_parametersD2 = material_parametersD2
        self.valve_parameters = valve_parameters
        self.environment_parameters = environment_parameters
        self.joint_parameters = joint_parameters
        self.simulation_parametersD1 = simulation_parametersD1
        self.simulation_parametersD2 = simulation_parametersD2
        self.pressure_inputDim = pressure_input_dim
        self.model_interpolation = model_interpolation
        self.device  = device
        self.scaler = scaler
        self.critic_iteration = critic_epochs
        self.mse_loss = nn.MSELoss() 
        self.l1_loss = nn.L1Loss()
        self.teqD1 = self.simulation_parametersD1["teq"]
        self.teqD2 = self.simulation_parametersD2["teq"]
        self.num_AS = 3
        self.num_TS = 6
        self.currentXeeStart = 12
        self.lambda_cycle = 10   
        self.lambda_gen = 1
        self.lambda_gp = 10
        self.lambda_error = 5
        
        self.Testbatch_size = testBatchSize
        self.testTrajDuration = 30
        self.modelMLPd1 = modelMLPd1.to(device)
        self.model_gen1 = GAN_model1.to(device)
        self.model_gen2 = GAN_model2.to(device)
        self.model_disc1 = DISC_model1.to(device)
        self.model_disc2 = DISC_model2.to(device)

        self.optimizerMLPd1 = torch.optim.Adam(self.modelMLPd1.parameters(), lr=self.learning_rate, 
                                                betas=(0.9,0.999), weight_decay=1e-5)
        # ## AdamW 
        self.opt_disc = AdamW(list(self.model_disc1.parameters()) + list(self.model_disc2.parameters()), 
                              lr=self.lr_generator, betas=(0.5,0.999))
        self.opt_gen = AdamW(list(self.model_gen1.parameters()) + list(self.model_gen2.parameters()), 
                                lr=self.lr_disc, betas=(0.5,0.999))
        self.errorEpoch = []
        self.MLP_train_lossArray = []
        self.DISC_train_lossArray = []
        self.GEN_train_lossArray = []
        self.val_lossArray = []


    def train(self, train_data, ukTestData, Isupport_sim, load_saved_model, shapeName):
        # Dummy test phase 
        if load_saved_model:
            customShape_test(ukTestData, Isupport_sim, shapeName, self.Testbatch_size, self.model_gen2, 10, 
                     1, self.numeric_parameters, self.geometric_parameters, self.material_parametersD2, 
                     self.valve_parameters, self.environment_parameters, self.simulation_parametersD2, self.material_parametersD1, 
                     self.simulation_parametersD1, self.teqD2, self.teqD1, self.device, is_val=False)
        # Train phase
        train_model(self.model_gen1, self.model_gen2, self.model_disc1, self.model_disc2, self.model_interpolation, self.critic_iteration,
                    self.epochs, self.opt_disc, self.opt_gen, self.l1_loss, self.mse_loss, self.batch_size, self.numeric_parameters, 
                    self.geometric_parameters, self.material_parametersD2, self.valve_parameters, self.environment_parameters, self.simulation_parametersD2,
                    self.material_parametersD1, self.simulation_parametersD1, self.teqD1, self.teqD2, train_data, ukTestData, 
                    Isupport_sim, self.device, load_saved_model)




def main():
    EPOCHS = 550 
    CRITIC_EPOCHS = 1
    LEARNING_RATE = 1e-4
    LEARNING_RATE_GENERATOR = 1e-4
    LEARNING_RATE_DISCRIMINATOR = 2e-4
    BATCH_SIZE = 10   
    TEST_BATCH_SIZE = 10
    LOAD_SAVED_MODEL = False
    model_interpolation = False
    addNoise = False
    noiseLevel = 0.5
    shapeName = "automatic_circle121"

    pressureIODim = 3
    generatorInputDim = 3
    d1MLP_InputDim = 30   # 6 previous pressure (t-1), 6 curr Xee(t), 6 prev Xee(t-1), 6 prev Xee(t-2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensorboard_path = "../trainings"
    ## 30sec
    pathActuationD1 = "../../../../../../Babbling_data/domain1/BPD1_1Mod_30sec.csv"
    pathTaskD1 = "../../../../../../Babbling_data/domain1/BTD1_1Mod_30sec.csv"    
    pathActuationD2 = "../../../../../../Babbling_data/domain2/BPD2_1Mod_visc8k_30sec.csv"
    pathTaskD2 = "../../../../../../Babbling_data/domain2/BTD2_1Mod_visc8k_30sec.csv"

    ## --------------- Custom shapes ---------------
    # ##Circle
    pathCS_circleD1 = "../../../../../../Babbling_data/domain1/circle60s_1Mod_TS.csv"
    pathCS_circleActuationD1 = "../../../../../../Babbling_data/domain1/circle60s_1Mod_AS.csv" 
    
    # ## DAmping circle
    # pathCS_circleD1 = "../../../../../../Babbling_data/domain1/circle_damp60s_1Mod_TS.csv"
    # pathCS_circleActuationD1 = "../../../../../../Babbling_data/domain1/circle_damp60s_1Mod_AS.csv"
    
    # # # #Rectangle
    # pathCS_circleD1 = "../../../../../../Babbling_data/domain1/rect60s_1Mod_TS.csv"
    # pathCS_circleActuationD1 = "../../../../../../Babbling_data/domain1/rect60s_1Mod_AS.csv"    

    ## Ellipse
    # pathCS_circleD1 = "../../../../../../Babbling_data/domain1/ellipse60s_1Mod_TS.csv"
    # pathCS_circleActuationD1 = "../../../../../../Babbling_data/domain1/ellipse60s_1Mod_AS.csv"    

    ## kite
    # pathCS_circleD1 = "../../../../../../Babbling_data/domain1/kite60s_1Mod_TS.csv"
    # pathCS_circleActuationD1 = "../../../../../../Babbling_data/domain1/kite60s_1Mod_AS.csv"       

    # --------------- Repeatation variance for different shapes ---------------
    # ## Circle
    # pathCS_circleD1 = "../../../../../../Babbling_data/domain1/circle5turns_TS.csv"
    # pathCS_circleActuationD1 = "../../../../../../Babbling_data/domain1/circle5turns_AS.csv"
 
    ## DAmping circle
    # pathCS_circleD1 = "../../../../../../Babbling_data/domain1/circle_damp5turns_TS.csv"
    # pathCS_circleActuationD1 = "../../../../../../Babbling_data/domain1/circle_damp5turns_AS.csv"   
    
    # # ## Rectangle
    # pathCS_circleD1 = "../../../../../../Babbling_data/domain1/rect5turns_TS.csv"
    # pathCS_circleActuationD1 = "../../../../../../Babbling_data/domain1/rect5turns_AS.csv"    

    modelMLP_d1 = MLPd1T1(input_dim=d1MLP_InputDim, out_dim=pressureIODim)
    disc_d1 = Discriminator(input_dim=1, output_dim=pressureIODim+6)
    disc_d2 = Discriminator(input_dim=1, output_dim=pressureIODim+6)
    gen_d1 = Generator(input_features=generatorInputDim+6, hidden_size= 32, num_layers=2, sequence_length=1, 
                                            batch_size=BATCH_SIZE, out_dim=pressureIODim+6, device=device)
    gen_d2 = Generator(input_features=generatorInputDim+6, hidden_size=32, num_layers=2, sequence_length=1, 
                                            batch_size=BATCH_SIZE, out_dim=pressureIODim+6, device=device)

    numeric_parameters, geometric_parameters, material_parametersD2, valve_parameters, \
    environment_parameters, simulation_parametersD2, joint_parameters = ISupportParamD2().paramaters()
    numeric_parameters, geometric_parameters, material_parametersD1, valve_parameters, \
    environment_parameters, simulation_parametersD1, joint_parameters = ISupportParamD1().paramaters()
    d1d2_train, customShapeCircle = processedData(pathActuationD1=pathActuationD1, 
                                                    pathTaskD1=pathTaskD1, pathActuationD2=pathActuationD2, 
                                                    pathTaskD2=pathTaskD2, pathCS_circle = pathCS_circleD1, 
                                                    pathCS_circleActuationD1=pathCS_circleActuationD1, addNoise=addNoise,
                                                    noiseLevel=noiseLevel)

    JT = JointTraining(modelMLPd1=modelMLP_d1,GAN_model1=gen_d1, GAN_model2=gen_d2, 
                    DISC_model1=disc_d1, DISC_model2=disc_d2, scaler=None, 
                    epochs = EPOCHS, critic_epochs = CRITIC_EPOCHS, learning_rate = LEARNING_RATE, 
                    lr_generator= LEARNING_RATE_GENERATOR, lr_disc=LEARNING_RATE_DISCRIMINATOR, 
                    batch_size = BATCH_SIZE, testBatchSize=TEST_BATCH_SIZE, patience = None, 
                    numeric_parameter = numeric_parameters, 
                    geometric_parameter = geometric_parameters, 
                    material_parametersD1 = material_parametersD1, 
                    material_parametersD2 = material_parametersD2, 
                    valve_parameters = valve_parameters, 
                    environment_parameters = environment_parameters, 
                    joint_parameters = joint_parameters, 
                    simulation_parametersD1 = simulation_parametersD1,
                    simulation_parametersD2=simulation_parametersD2, 
                    pressure_input_dim = pressureIODim, model_interpolation = model_interpolation, 
                    device=device)

    ## Empty the tensorboard logs
    removeLogs = input("Do you want to remove the previous tensorboard logs? Y/N")
    if removeLogs == "Y" or removeLogs == "y": 
        if os.path.isdir(tensorboard_path):  
            ## Remove the trainings directory
            shutil.rmtree(tensorboard_path)
            print("Logs removed!!") 
            JT.train(train_data=d1d2_train, ukTestData=customShapeCircle, 
                    Isupport_sim=ISupportElastica, 
                    load_saved_model=LOAD_SAVED_MODEL, 
                    shapeName=shapeName)

        else:
            JT.train(train_data=d1d2_train, ukTestData=customShapeCircle, 
                     Isupport_sim=ISupportElastica, 
                     load_saved_model=LOAD_SAVED_MODEL, 
                     shapeName=shapeName)

    
    elif removeLogs == "N" or removeLogs == "n":
        JT.train(train_data=d1d2_train, ukTestData=customShapeCircle, 
                 Isupport_sim=ISupportElastica, 
                 load_saved_model=LOAD_SAVED_MODEL, 
                 shapeName=shapeName)

    
    else:
        print("Incorrect!! Option")
        main()  

if __name__=="__main__":
    main()