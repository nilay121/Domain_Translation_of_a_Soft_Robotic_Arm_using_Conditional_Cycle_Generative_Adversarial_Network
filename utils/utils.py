import torch 
import pickle
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.utils.optimize import _check_optimize_result
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline


class CustomDatasetForDataLoader(Dataset):
    def __init__(self,data,targets):
        self.data = data
        self.targets = targets
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],self.targets[idx]
    
def processedData(pathActuationD1, pathTaskD1, pathActuationD2, pathTaskD2, pathCS_circle, 
                  pathCS_circleActuationD1, addNoise=False, noiseLevel=0.1):
    TEST_SIZE = 0.20
    ## Domain1 
    actuationD1 = pd.read_csv(pathActuationD1, delimiter=",")
    eeCoordinatesD1 = pd.read_csv(pathTaskD1, delimiter=",")   
    
    ##Domain2
    actuationD2 = pd.read_csv(pathActuationD2, delimiter=",")        
    eeCoordinatesD2 = pd.read_csv(pathTaskD2, delimiter=",")
    ## Custom circle

    eeCordinate_circleD1 = pd.read_csv(pathCS_circle, delimiter=",")
    actuation_circleD1 = pd.read_csv(pathCS_circleActuationD1, delimiter=",")

    eeCoordinatesD1 = np.concatenate([eeCoordinatesD2, eeCoordinatesD1], axis=1, dtype="float32")                          
    featuresXee = eeCoordinatesD1

    ## Combining the actuation for both domains
    targetPressure = np.concatenate([np.array(actuationD2, dtype="float32"), 
                                     np.array(actuationD1, dtype="float32")], axis=1) 
    
    Xee_CSD1 =  np.array(eeCordinate_circleD1, dtype="float32")[0:600]
    actuation_CSD1 = np.array(actuation_circleD1, dtype="float32")[0:600]

    ## Add noise to the custom actuation values
    if addNoise:
        actuation_CSD1 = (actuation_CSD1 + noiseLevel*np.random.normal(loc=1, scale=0.5, size=(actuation_CSD1.shape[0], actuation_CSD1.shape[1]))).astype("float32")

    ## Create Val dataset
    val_idx = int(TEST_SIZE*featuresXee.shape[0])
    trainFeaturesXee = featuresXee[0:featuresXee.shape[0]-(val_idx)]
    trainTargetsPressure = targetPressure[0:targetPressure.shape[0]-(val_idx)]
    valFeaturesXee = featuresXee[featuresXee.shape[0]-val_idx:]
    valTargetsPressure = targetPressure[targetPressure.shape[0]-val_idx:]
    d1d2_train = CustomDatasetForDataLoader(data=trainFeaturesXee, targets=trainTargetsPressure)
    d1d2_test = CustomDatasetForDataLoader(data=valFeaturesXee, targets=valTargetsPressure)
    customShapeCircle = CustomDatasetForDataLoader(data=Xee_CSD1, targets=actuation_CSD1)

    # ## unknown test data
    print(f"val Xee d1 min max {valFeaturesXee.min()}, {valFeaturesXee.max()}")
    print(f"val Pressure d2 min max {valTargetsPressure.min()}, {valTargetsPressure.max()}")
    print(f"train Xee d1 min max {trainFeaturesXee.min()}, {trainFeaturesXee.max()}")
    print(f"train Pressure d2 min max {trainTargetsPressure.min()}, {trainTargetsPressure.max()}")

    return d1d2_train, customShapeCircle

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, S, F = real.shape
    alpha = torch.rand((BATCH_SIZE, S, F), requires_grad=True).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]  ## Extract first element
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def interpolate_actuation(actuation_pressure, taskSpace_traj, interpolatedDataLen):
    '''
    3000 ---> 5 times
    2400 ---> almost 4 times
    1800 ---> 3 times --> We use this
    '''
    data_len = actuation_pressure.shape[0]
    t_steps = np.arange(0, data_len) / data_len
    t_stepsInterpolation = np.arange(0, interpolatedDataLen) / interpolatedDataLen
    cs_AS = CubicSpline(t_steps, actuation_pressure) 
    cs_TS = CubicSpline(t_steps, taskSpace_traj)
    interpolated_pressure = cs_AS(t_stepsInterpolation).astype("float32")
    interpolated_trajectory = cs_TS(t_stepsInterpolation).astype("float32")

    return interpolated_pressure, interpolated_trajectory

def sampleFromInterpolatedData(interpolated_predTraj, interpolated_actualtraj, modCheckerIdx):
    sampledPred_traj = []
    sampedActual_traj = []
    for i in range(interpolated_predTraj.shape[0]):
        if i%modCheckerIdx == 0:
            sampledPred_traj.append(interpolated_predTraj[i])
            if interpolated_actualtraj is not None:
                sampedActual_traj.append(interpolated_actualtraj[i]) 
    sampedActual_traj = np.array(sampedActual_traj)
    sampledPred_traj = np.array(sampledPred_traj)
    return sampedActual_traj, sampledPred_traj

def addFromInterpolatedData(interpolated_predTraj, modCheckerIdx):
    sampledPred_traj = []
    for i in range(interpolated_predTraj.shape[0]):
        if (i%modCheckerIdx == 0):
            if i == 0:
                sampledPred_traj.append(np.sum(interpolated_predTraj[i-modCheckerIdx:i], axis=0))
            else:
                sampledPred_traj.append(interpolated_predTraj[i])
    sampledPred_traj = np.array(sampledPred_traj)
    return sampledPred_traj

def orientationError(actual, pred):
    eps = 1e-7
    theta = np.arccos(np.sum(actual*pred, axis=1)/(np.linalg.norm(actual, \
                                    axis=1)*np.linalg.norm(pred, axis=1) + eps))
    return theta   

def save_dictionary(data, epoch):
    with open(f'interpolation_steps_{epoch}.pkl', 'wb') as fp:
        pickle.dump(data, fp)
        print('dictionary saved successfully to file')

def load_dictionary():
    with open('interpolation_steps.pkl', 'rb') as fp:
        interpolation_steps = pickle.load(fp)
    return interpolation_steps

# def inverseTransform(data):
#     return ((data*0.5)+0.5)*3.5

def PressurePlotter(pred_data, ground_truth, valveNumb, domain):
    x = [i for i in range (len(pred_data[0:]))]
    plt.plot(x, pred_data[:,valveNumb], color='navy', alpha=0.7)
    plt.plot(x, ground_truth[:,valveNumb], color='green', alpha=0.5)
    plt.title("Original and Predicted pressure")
    plt.xlabel("Time")
    plt.ylabel("Pressure values")
    plt.legend(['Pred Data','Ground truth'], loc=4)
    plt.savefig(f"../images/{domain}PressureValve{valveNumb}.png")
    plt.show()
    plt.clf()

def PressurePlotterCS(pred_dataD1, pred_dataD2, ground_truth, valveNumb, domain):
    x = [i for i in range (len(pred_dataD2[0:]))]
    plt.plot(x, pred_dataD1[:,valveNumb], color='red', alpha=0.6)
    plt.plot(x, pred_dataD2[:,valveNumb], color='navy', alpha=0.7)
    plt.plot(x, ground_truth[:,valveNumb], color='green', alpha=0.5)
    plt.title("Original and Predicted pressure")
    plt.xlabel("Time")
    plt.ylabel("Pressure values")
    plt.legend(['Pred MLP D1', 'Pred GAN D2', 'Ground truth'], loc=4)
    plt.savefig(f"../images/{domain}PressureValve{valveNumb}.png")
    plt.show()
    plt.clf()

def TSTime(pred_data, ground_truthD2, x, y, z, shape):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,5))
    axes[0].plot(pred_data[:, x], color="green")
    axes[0].plot(ground_truthD2[:, x], color="navy")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("End Effector X-axis")
    axes[0].grid()
    axes[0].set_title("Prediction vs Target of End Effector")
    axes[0].legend(["Prediction", "Target"])

    axes[1].plot(pred_data[:, y], color="green")
    axes[1].plot(ground_truthD2[:, y], color="navy")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("End Effector Y-axis")
    axes[1].grid()

    axes[2].plot(pred_data[:, z], color="green")
    axes[2].plot(ground_truthD2[:, z], color="navy")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("End Effector Z-axis")
    axes[2].grid()
    plt.tight_layout()
    plt.savefig(f"../images/trajWithTime{shape}.png")
    plt.show()
    plt.clf() 

def ErrorTime(errorXee, shape):
    plt.plot(errorXee,"b--", alpha=0.7)
    plt.title("Original vs Predicted task space")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.savefig(f"../images/errorWithTime{shape}.png")
    plt.show()
    plt.clf()    

def TaskSpacePlotter(pred_data, ground_truthD2, x_idx, y_idx, shape):
    # plt.plot(pred_data[:,x_idx], pred_data[:,y_idx], color="navy", alpha=0.7)
    # plt.plot(ground_truthD2[:,x_idx], ground_truthD2[:,y_idx], color='orange', alpha=0.5)
    plt.scatter(pred_data[:,x_idx], pred_data[:,y_idx], color="navy", alpha=0.7)
    plt.scatter(ground_truthD2[:,x_idx], ground_truthD2[:,y_idx], color='orange', alpha=0.5)
    plt.title("Original vs Predicted task space")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend(['Pred Data', 'Ground truth'], loc=4)
    plt.savefig(f"../images/XY_plot{shape}.png")
    plt.show()
    plt.clf()


def TaskSpacePlotterAllpred(pred_dataf2d2, ground_truthD2, pred_dataf2d1, x_idx, y_idx, shape):
    plt.scatter(pred_dataf2d2[:,x_idx], pred_dataf2d2[:,y_idx], color="navy", alpha=0.7)
    plt.scatter(ground_truthD2[:,x_idx], ground_truthD2[:,y_idx], color='orange', alpha=0.5)
    plt.scatter(pred_dataf2d1[:,x_idx], pred_dataf2d1[:,y_idx], color='green', alpha=0.1)
    plt.title("Original vs Predicted task space")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # plt.legend(['FM2D2', 'Ground truth', "FM1D1", "FM2D1"], loc=4)
    plt.legend(['FM2D2','Ground truth', "FM1D1", "FM2D1"], loc=4)
    plt.savefig(f"../images/XY_plot{shape}.png")
    plt.show()
    plt.clf()

def plotTrainVal_loss(dataTrain):
    epochs = [i for i in range(np.array(dataTrain).shape[0])]
    plt.plot(epochs, dataTrain)
    # plt.plot(epochs, dataVal)
    plt.xlabel("Epoch")
    plt.ylabel("loss value")
    plt.legend(["Train loss"])
    plt.savefig(f"../images/loss_plot.png")
    plt.show()
    plt.clf()

def plotGAN_loss(lossTrain, title):
    epochs = [i for i in range(np.array(lossTrain).shape[0])]
    plt.plot(epochs, lossTrain)
    plt.xlabel("Epoch")
    plt.ylabel("loss value")
    plt.title(title)
    plt.savefig(f"../images/{title}loss_plot.png")
    plt.show()
    plt.clf()

def plot3dTrajectory(pred_data, ground_truth, shape):
    ax = plt.figure().add_subplot(projection = "3d")
    ax.plot(pred_data[:,0], pred_data[:,1], pred_data[:,2], color="red")
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], color="navy", alpha=0.3)
    # ax.title("Original vs Predicted 3d task space")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend(['Pred Data', 'Ground truth'], loc=4)
    plt.savefig(f"../images/3dplot{shape}.png")
    plt.show()
    plt.clf()