
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
                        sampleFromInterpolatedData , save_dictionary
from train_test_utils import closedLoopModelPred, reset_isupport_model, \
                            deterministic_robotModel, interpolation_steps_calc, \
                            metricsReport, save_logs, save_errors
from torch.utils.tensorboard import SummaryWriter


def train_model(model_gen1, model_gen2, model_disc1, model_disc2, model_interpolation,
          critic_iteration, epochs, opt_disc, opt_gen, l1_loss, mse_loss, batch_size, 
          numeric_parameters, geometric_parameters, material_parametersD2, 
          valve_parameters, environment_parameters, simulation_parametersD2,
          material_parametersD1, simulation_parametersD1, teqD1, 
          teqD2, train_data, ukTestData, Isupport_sim, device, load_saved_model=False):
    
    skip = True
    global_step = 0
    lower_interpolate_step = 5
    upper_interpolate_step = 35
    domain2TS_idx = 6
    domain2AS_idx = 3
    seq_length = 1
    lambda_cycle = 10   
    lambda_gen = 1
    lambda_gp = 10
    lambda_error = 5

    interpolation_info = {}

    writer_ganLoss = SummaryWriter(log_dir = "../trainings/gan_loss")
    writer_discLoss = SummaryWriter(log_dir = "../trainings/disc_loss")
    writer_valCS = SummaryWriter(log_dir = "../trainings/val_cs")
    load_saved_model = load_saved_model
    
    if load_saved_model and model_interpolation:
        print("Loading the trained model with self interpolation..........")
        model_gen1 = torch.load("../saved_models_good_automatic/gen1ModelD1Epoch150.pickle")
        model_gen2 = torch.load("../saved_models_good_automatic/gen2ModelD2Epoch150.pickle")
    elif load_saved_model:
        print("Loading the trained model..........")
        model_gen1 = torch.load("../saved_models_good_automatic/gen1ModelD1Epoch150.pickle")
        model_gen2 = torch.load("../saved_models_good_automatic/gen2ModelD2Epoch150.pickle")          
    else:
        print("Starting the training process..............")
        train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=False)
        model_disc1.train()
        model_disc2.train()
        model_gen1.train()
        model_gen2.train()

        for epoch in range(epochs):
            train_lossGEN = 0
            train_lossDISC = 0
            totalbatches = 0
            val_loss = 0
            
            loader_loop = tqdm(train_data_loader, leave=True)
            flag_first = True

            if epochs%10 == 0:
                skip = False
            
            # loop through minibatches 
            for batch_idx, (Xee, pressureCombined) in enumerate(loader_loop):
                ## domain2 trajectories
                XeeD2ForGAN = Xee[:, :domain2TS_idx].to(device)   
                ## domain1 trajectories --> it contains the Xee at time t, t-1, t-2
                XeeD1ForGAN = Xee[:, domain2TS_idx:].to(device) 
                    ## domain2 pressure values  
                pressureD2 = pressureCombined[:, :domain2AS_idx].to(device)   
                ## domain1 pressure values
                pressureD1 = pressureCombined[:, domain2AS_idx:].to(device) 
                current_batch = pressureD2.shape[0]   
                
                # save one trajectory for interpolation calculations
                # if len(traj_collector_d1) < 300:
                #     tempXeeD1 = XeeD1ForGAN.detach().cpu().numpy()
                #     tempPressureD1 = pressureD1.detach().cpu().numpy()
                #     if len(traj_collector_d1) == 0:                            
                #         traj_collector_d1.append(tempXeeD1)
                #         pressure_collector_d1.append(tempPressureD1)
                #         traj_collector_d1 = np.array(traj_collector_d1).reshape(-1, self.num_TS)
                #         pressure_collector_d1 = np.array(pressure_collector_d1).reshape(-1, self.num_AS)
                #     else:
                #         traj_collector_d1 = np.concatenate((traj_collector_d1, tempXeeD1), axis=0)
                #         pressure_collector_d1 = np.concatenate((pressure_collector_d1, tempPressureD1), axis=0)
                # Interpolation steps calculation

                if (epoch == 0):
                    if not skip:
                        interpolation_steps = interpolation_steps_calc(lower_interpolate_step, upper_interpolate_step, Isupport_sim, seq_length, 
                                                                    numeric_parameters, geometric_parameters, material_parametersD2, 
                                                                    valve_parameters, environment_parameters, simulation_parametersD2,
                                                                    model_gen2, teqD2, mse_loss, interpolation_info, device)
                        # interpolation_steps = 15
                        skip = True
                        # save the dictionary
                        save_dictionary(interpolation_info, epoch)
                
                # Previous trajectory (t-1) for forward model input
                if flag_first:
                    prev_XeeD2 = torch.tile(torch.tensor([0,0,-0.38,0,0,-1]), (current_batch, 1)).to(device)
                else:
                    prev_XeeD2 = pred_fm2d2  
                ## Discriminator training
                for _ in range(critic_iteration):
                    ## GAN 2
                    GAN2_inputPressure = torch.cat([pressureD1, XeeD1ForGAN], axis=1).reshape(current_batch, 
                                                    seq_length, -1).to(device) 
                    # GAN2 predictions --> domain2 pressure values 
                    GANpredPressure_d2 = model_gen2(GAN2_inputPressure, False) 
                    ## GAN 1         
                    GAN1_inputPressure = torch.cat([pressureD2, XeeD2ForGAN], axis=1, ).reshape(current_batch, 
                                                    seq_length, -1).to(device)   
                    # GAN1 predictions --> domain1 pressure values
                    GANpredPressure_d1 = model_gen1(GAN1_inputPressure, False)    

                    ## Disc 2
                    D_d2_real = model_disc2(x = torch.cat([pressureD2, XeeD2ForGAN], axis=1).reshape(current_batch, 
                                                    seq_length, -1)).reshape(-1)   # Real data
                    D_d2_fake = model_disc2(x = GANpredPressure_d2.detach().reshape(GANpredPressure_d2.shape[0], 
                                                    seq_length,-1)).reshape(-1)    # Fake data
                    D_d2_gp = gradient_penalty(critic = model_disc2, real = torch.cat([pressureD2, XeeD2ForGAN], 
                                                    axis = 1).reshape(current_batch, seq_length, -1), 
                                                    fake = GANpredPressure_d2.detach().reshape(current_batch, 
                                                    seq_length,-1), device = device)
                    D_d2_CombLoss = -(torch.mean(D_d2_real) - torch.mean(D_d2_fake)) + lambda_gp*D_d2_gp
                    
                    ## Disc 1
                    D_d1_real = model_disc1(x = torch.cat([pressureD1, XeeD1ForGAN], axis=1).reshape(current_batch, 
                                                    seq_length, -1)).reshape(pressureD1.shape[0], -1)  # Real data
                    D_d1_fake = model_disc1(x = GANpredPressure_d1.detach().reshape(current_batch, 
                                                    seq_length, -1)).reshape(GANpredPressure_d1.shape[0], -1) # Fake data
                    D_d1_gp = gradient_penalty(critic = model_disc1, real = torch.cat([pressureD1, XeeD1ForGAN], 
                                                    axis = 1).reshape(current_batch, seq_length, -1),
                                                    fake = GANpredPressure_d1.detach().reshape(current_batch, 
                                                    seq_length, -1), device = device)
                    D_d1_CombLoss = -(torch.mean(D_d1_real) - torch.mean(D_d1_fake)) + lambda_gp*D_d1_gp

                    ## Total discriminator loss
                    totalDisc_loss = D_d2_CombLoss + D_d1_CombLoss
                    model_disc1.zero_grad()
                    model_disc2.zero_grad()  
                    totalDisc_loss.backward(retain_graph=True)
                    opt_disc.step()
                
                ## Forward model prediction for closed loop and loss calculation
                fm2d2_input = torch.concat([prev_XeeD2[-(batch_size-current_batch):], GANpredPressure_d2[:, 0:3]/3.5], 
                                            axis=1).reshape(current_batch, seq_length, -1)
                pred_fm2d2 = closedLoopModelPred(fm2d2_input, device)  
                loss_error = mse_loss(pred_fm2d2, XeeD2ForGAN)
                
                ## Generator training
                gen2_d2_fake = model_disc2(GANpredPressure_d2.reshape(GANpredPressure_d2.shape[0], 1, 
                                                GANpredPressure_d2.shape[1])).reshape(-1)
                gen1_d1_fake = model_disc1(GANpredPressure_d1.reshape(GANpredPressure_d1.shape[0], 1, 
                                                GANpredPressure_d1.shape[1])).reshape(-1)
                loss_G_d2 = -torch.mean(gen2_d2_fake)  ### is it coorect?? -ve sign needed in CGAN with GP??
                loss_G_d1 = -torch.mean(gen1_d1_fake)
                
                ## GAN1 cycle consistency
                cycleInputGAN1 = GANpredPressure_d2.reshape(GANpredPressure_d2.shape[0], seq_length, -1).to(device)
                cycle_d1 = model_gen1(cycleInputGAN1, False)
                
                # GAN2 cycle consistency
                cycleInputGAN2 = GANpredPressure_d1.reshape(GANpredPressure_d1.shape[0], seq_length, -1).to(device)                        
                cycle_d2 = model_gen2(cycleInputGAN2, False)

                cycle_d1_loss = l1_loss(torch.cat([pressureD1, XeeD1ForGAN], axis=1), cycle_d1)
                cycle_d2_loss = l1_loss(torch.cat([pressureD2, XeeD2ForGAN], axis=1), cycle_d2)
                
                ## GAN losses
                loss_gen = (
                        loss_G_d2 * lambda_gen +
                        loss_G_d1 * lambda_gen + 
                        cycle_d1_loss * lambda_cycle +
                        cycle_d2_loss * lambda_cycle +
                        loss_error*lambda_error
                        ) 
                
                model_gen1.zero_grad()
                model_gen2.zero_grad()# tes the 
                loss_gen.backward()
                opt_gen.step()

                train_lossDISC += totalDisc_loss.item() 
                train_lossGEN += loss_gen.item()
                totalbatches += 1

                ## display results
                loader_loop.set_description(f"Epoch {epoch}/{epochs}")
                loader_loop.set_postfix_str(f"Batch {batch_idx}/{len(loader_loop)} lossG {loss_gen.item():.4f} lossDIsc {totalDisc_loss.item():.4f} error {loss_error.item():.4f}")
            
            ## Update the GAN and disc loss to tensorboard every epoch
            writer_ganLoss.add_scalar("GAN_loss", loss_gen.item(), epochs)
            writer_discLoss.add_scalar("Disc_loss", totalDisc_loss.item(), epochs)   
            
            ## Display the trajectory to tensorboard after every 50 epochs
            if (epoch%50 == 0):
                plot_to_tensorboard(ukTestData, Isupport_sim, writer_valCS, None, 10, 
                    model_gen2, interpolation_steps, seq_length, numeric_parameters, geometric_parameters, 
                    material_parametersD2, valve_parameters, environment_parameters, simulation_parametersD2, 
                    material_parametersD1, simulation_parametersD1, teqD2, teqD1, device, is_val=True)

                ## save the mlp and gen models
                torch.save(model_gen2, f"../saved_models/gen2ModelD2Epoch{epoch}.pickle")
                torch.save(model_gen1, f"../saved_models/gen1ModelD1Epoch{epoch}.pickle")           

            GEN_train_lossArray.append(train_lossGEN/totalbatches)
            DISC_train_lossArray.append(train_lossDISC/totalbatches)

        ## Plot train val loss 
        GEN_train_lossArray = np.array(GEN_train_lossArray)
        DISC_train_lossArray = np.array(DISC_train_lossArray)

        plotGAN_loss(GEN_train_lossArray, title="Generator loss")
        plotGAN_loss(DISC_train_lossArray, title="Discriminator loss")

        ## save the mlp and gen models
        torch.save(model_gen2, "../saved_models/gen2ModelD2.pickle")
        torch.save(model_gen1, "../saved_models/gen1ModelD1.pickle")

    
def plot_to_tensorboard(ukTestData, Isupport_sim, writer_valCS, shapeName, Testbatch_size, 
                    model_gen2, interpolation_steps, seq_length, numeric_parameters, geometric_parameters, 
                    material_parametersD2, valve_parameters, environment_parameters, simulation_parametersD2, 
                    material_parametersD1, simulation_parametersD1, teqD2, teqD1, device, is_val=True):

    fm2d2_pred, orgXee_pred, f2d1_pred = customShape_test(ukTestData, Isupport_sim, shapeName, Testbatch_size, 
                    model_gen2, interpolation_steps, seq_length, numeric_parameters, geometric_parameters, 
                    material_parametersD2, valve_parameters, environment_parameters, simulation_parametersD2, 
                    material_parametersD1, simulation_parametersD1, teqD2, teqD1, device, is_val=is_val)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,5))
    axes[0].scatter(fm2d2_pred[:, 0], fm2d2_pred[:, 1], alpha=0.5, color="navy")
    axes[0].scatter(orgXee_pred[:, 0], orgXee_pred[:, 1], alpha=0.2, color="green")
    axes[0].scatter(f2d1_pred[:, 0], f2d1_pred[:, 1], alpha=0.1, color="yellow")
    axes[0].set_xlabel("X-axis")
    axes[0].set_ylabel("Y-axis")
    axes[0].legend(["FM2D2", "Org Xee", "FM2D1"])
    axes[0].grid()

    axes[1].plot(fm2d2_pred[:, 0], color="navy", linestyle="-")
    axes[1].plot(fm2d2_pred[:, 1], color="navy", linestyle=":")
    axes[1].plot(fm2d2_pred[:, 2], color="navy", linestyle="-.", label="FM2D2")
    axes[1].plot(orgXee_pred[:, 0], color="green", linestyle="-")
    axes[1].plot(orgXee_pred[:, 1], color="green", linestyle=":")
    axes[1].plot(orgXee_pred[:, 2], color="green", linestyle="-.", label="Org Xee")
    axes[1].set_xlabel("Trajectory")
    axes[1].set_ylabel("Time")
    axes[1].legend()
    axes[1].grid()                    
    plt.tight_layout()
    writer_valCS.add_figure("Val_CS", plt.gcf(), global_step=global_step)
    global_step+=1
    writer_valCS.close()  



def customShape_test(ukTestData, Isupport_sim, shapeName, Testbatch_size, model_gen2, interpolation_steps,
                     seq_length, numeric_parameters, geometric_parameters, material_parametersD2, 
                     valve_parameters, environment_parameters, simulation_parametersD2, material_parametersD1, 
                     simulation_parametersD1, teqD2, teqD1, device, is_val=False):
    
    OrgXeeD1 = predPressureD1 = predPressureD2 = []
    error_plot = predXeeF2D2 = predXeeFM2D1 = predXeeFM1D1 = []

    test_data_loader = DataLoader(dataset=ukTestData, batch_size=Testbatch_size, shuffle=False)
    loader_loop = tqdm(test_data_loader,leave=True)
    model_gen2.eval()

    ## Initialize to rest position 
    isupportFM2D2 = reset_isupport_model(Isupport_sim, numeric_parameters, geometric_parameters, material_parametersD2, 
                                valve_parameters, environment_parameters, simulation_parametersD2)
    isupportFM2D1 = reset_isupport_model(Isupport_sim, numeric_parameters, geometric_parameters, material_parametersD2, 
                                valve_parameters, environment_parameters, simulation_parametersD2)
    isupportFM1D1 = reset_isupport_model(Isupport_sim, numeric_parameters, geometric_parameters, material_parametersD1, 
                                valve_parameters, environment_parameters, simulation_parametersD1)
    
    with torch.no_grad():

        for Xee, actuations in loader_loop:

            Xee_d1 = Xee.to(device)
            pressureD1 = actuations.to(device)  
            # gen2 predictions   
            gen2_InputPressured1 = torch.cat([pressureD1, Xee_d1], axis=1).reshape(pressureD1.shape[0], 
                                                                                seq_length, -1)  
            gen2_pred = model_gen2(gen2_InputPressured1, True, interpolation_steps)

            ## GAN predicted pressure                
            gen2_predPressure_d2 = gen2_pred[:,0:3]    
            # send to numpy
            predPressure_d1_sampled = pressureD1.detach().cpu().numpy()
            gen2_predPressure_d2Inv = gen2_predPressure_d2.detach().cpu().numpy()

            ## Forward elastica Model
            pred_combinedPosFM2D2 = deterministic_robotModel(Isupport_sim=isupportFM2D2, 
                                                        teq=teqD2, 
                                                        predictedPressure=gen2_predPressure_d2Inv)
            pred_combinedPosFM2D1 = deterministic_robotModel(Isupport_sim=isupportFM2D1, 
                                                        teq=teqD2,
                                                        predictedPressure=predPressure_d1_sampled)
            pred_combinedPosFM1D1 = deterministic_robotModel(Isupport_sim=isupportFM1D1,
                                                        teq=teqD1, 
                                                        predictedPressure=predPressure_d1_sampled)
            Xee_d1 = Xee_d1.detach().cpu().numpy()
            gen2_predPressure_d2Inv, pred_combinedPosFM2D2_sampled = sampleFromInterpolatedData(pred_combinedPosFM2D2, 
                                                                                                gen2_predPressure_d2Inv, 
                                                                                                interpolation_steps)
            ## display error
            error = np.linalg.norm((pred_combinedPosFM2D2_sampled[:, 0:3] - Xee_d1[:, 0:3]), axis=1)
                            
            ## save data to plot
            if len(predXeeF2D2) <= 0:
                predXeeF2D2.append(pred_combinedPosFM2D2)
                OrgXeeD1.append(Xee_d1)
                predPressureD2.append(gen2_predPressure_d2Inv.reshape(-1, 3))

                predXeeFM2D1.append(pred_combinedPosFM2D1)
                predXeeFM1D1.append(pred_combinedPosFM1D1)
                error_plot.append(error)
            else:

                predPressureD2 = np.concatenate([np.array(predPressureD2).reshape(-1, 3), 
                                            gen2_predPressure_d2Inv.reshape(-1, 3)], axis = 0)
                predXeeF2D2 = np.concatenate([np.array(predXeeF2D2).reshape(-1,6), 
                                            pred_combinedPosFM2D2], axis = 0)
                predXeeFM2D1 = np.concatenate([np.array(predXeeFM2D1).reshape(-1,6), 
                                            pred_combinedPosFM2D1], axis = 0)
                predXeeFM1D1 = np.concatenate([np.array(predXeeFM1D1).reshape(-1,6), 
                                            pred_combinedPosFM1D1], axis = 0)
                OrgXeeD1 = np.concatenate([np.array(OrgXeeD1).reshape(-1,6), Xee_d1], axis=0)
                error_plot = np.concatenate([np.array(error_plot).reshape(-1), error.reshape(-1)], 
                                            axis = 0)
                
    OrgXeeD1, predXeeF2D2  = sampleFromInterpolatedData(predXeeF2D2, OrgXeeD1, interpolation_steps)

    metricsReport(predPressureD1, predPressureD2, OrgXeeD1, predXeeF2D2, predXeeFM2D1,
                predXeeFM1D1, shapeName)
    
    if is_val:
        model_gen2.train()
        return predXeeF2D2, OrgXeeD1, predXeeFM2D1
    else:
        ## Save predictions to a text file
        save_logs(originalXeeD1 = OrgXeeD1, MLP_predXeeD2 = None, GAN_predXeeD2 = predXeeF2D2, 
                        XeeD2_WithD1 = predXeeFM2D1, shape=shapeName)

        TaskSpacePlotterAllpred(pred_dataf2d2=predXeeF2D2, ground_truthD2=OrgXeeD1,  
                                pred_dataf2d1=predXeeFM2D1, x_idx=0, y_idx=1, 
                                shape="CS_F2D2F1D1F2D1_TP_Carlo")
        TaskSpacePlotterAllpred(pred_dataf2d2=predXeeF2D2, ground_truthD2=OrgXeeD1, 
                                pred_dataf2d1=predXeeFM2D1, x_idx=3, y_idx=4, 
                                shape="CS_FM2D2FM1D1F2D1_DP_Carlo")
        TaskSpacePlotter(pred_data=predXeeFM2D1, ground_truthD2=OrgXeeD1, 
                            x_idx=3, y_idx=4, shape="CS_FM2D1_DP_Carlo")
        TaskSpacePlotter(pred_data=predXeeFM2D1, ground_truthD2=OrgXeeD1, 
                            x_idx=0, y_idx=1, shape="CS_FM2D1_TP_Carlo")
        TSTime(pred_data=predXeeF2D2, ground_truthD2=OrgXeeD1, 
                x=3, y=4, z=5, shape="TO_carlo")
        TSTime(pred_data=predXeeF2D2, ground_truthD2=OrgXeeD1, 
                x=0, y=1, z=2, shape="TP_carlo")
        ErrorTime(errorXee=error_plot, shape="CustomShape_carlo")
        plot3dTrajectory(pred_data=predXeeF2D2, ground_truth=OrgXeeD1, 
                            shape="CS_GANpred_carlo3d")


