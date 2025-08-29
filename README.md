![ccgan_gif](https://github.com/user-attachments/assets/75754dd6-1abe-43fd-8821-9c6b154890a8)# Soft-robotic-domain-translation-using-CCGAN

## Abstract
Deep learning provides a powerful method for modeling the dynamics of soft robots, offering advantages over traditional analytical approaches that require precise knowledge
of the robot’s structure, material properties, and other physical
characteristics. Given the inherent complexity and non-linearity
of these systems, extracting such details can be challenging. The
mappings learned in one domain cannot be directly transferred
to another domain with different physical properties. This chal-
lenge is particularly relevant for soft robots, as their materials
gradually degrade over time. In this paper, we introduce a
domain translation framework based on a conditional cycle
generative adversarial network (CCGAN) to enable knowledge
transfer from a source domain to a target domain. Specifically, we
employ a dynamic learning approach to adapt a pose controller
trained in a standard simulation environment to a domain with
tenfold increased viscosity. Our model learns from input pressure
signals conditioned on corresponding end-effector positions and
orientations in both domains. We evaluate our approach through
trajectory-tracking experiments across five distinct shapes and
further assess its robustness under noise perturbations and periodicity tests. The results demonstrate that CCGAN-GP effectively
facilitates cross-domain skill transfer, paving the way for more
adaptable and generalizable soft robotic controllers.

## Model Architecture and Working
![](https://github.com/nilay121/Soft-robotic-domain-translation-using-CCGAN/blob/main/ccgan_gif.gif)


## To Run the algorithm
- Create a virtual conda environment
  ```bash
  conda env create -f elastica_2022.yml
  ```
- Activate the virtual environmnet
  ```bash
  conda activate elastica_2022
  ```
- Move to the directory
  
- Attach the pyelastica simulator or any other simulator you want
  
- Then just run the main script
  ```bash
  python3 main_CGANWGP_bothXee_pressure.py
  ```
- Change the path for the actuations and the corresponding end-effector poses
- Train the cycle GAN network with the default parameters or change it if required

## To cite the paper
  ```bash
  @misc{kushawaha2025domaintranslationsoftrobotic,
        title={Domain Translation of a Soft Robotic Arm using Conditional Cycle Generative Adversarial Network}, 
        author={Nilay Kushawaha and Carlo Alessi and Lorenzo Fruzzetti and Egidio Falotico},
        year={2025},
        eprint={2508.14100},
        archivePrefix={arXiv},
        primaryClass={cs.RO},
        url={https://arxiv.org/abs/2508.14100}, 
  }
  ```
