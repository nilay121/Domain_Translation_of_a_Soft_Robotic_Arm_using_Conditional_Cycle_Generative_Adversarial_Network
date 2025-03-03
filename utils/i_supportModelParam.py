import sys
sys.path.append('../')
import numpy as np

## Domain 1 Deterministic Model
class ISupportParamD1():
    def __init__(self):
        pass
    def paramaters(self):
        # environment parameters
        g = 9.80513  # gravity in Pisa

        g_vector = np.array([0.0, -g, 0.0])  # (yzx)

        # geometric parameters, spatial reference=(y,z,x)
        start     = np.array([0.0,  0.0, 0.0])       # rod location
        direction = np.array([0.0, -1.0, 0.0])       # rod orientation
        normal    = np.array([0.0,  0.0, -1.0])      # rod normal direction [0,0,-1]

        module_length  = 0.19 # rod length [m]
        section_radius = 0.03 # rod radius (section) [m]
        delta          = 0.02 # distance between section centre and actuator centre [m]
        actuator_outer_radius = 0.00779 # media outer
        actuator_wall_thickness = 0.0014 # wall thickness of 3D printed actuator (1.4 mm)

        start_2 = start + direction * module_length

        # material parameters
        weight         = 0.1704 # (~0.181) total_weight                # manipulator weight          [Kg]       (180 g)
        density        = 1104   # average manipulator density [Kg]/[m]^3
        young_modulus  = 1.646439e+06  ###1.646439e+06--------------> real value
        poisson_ratio  = 0.5
        translational_damping_constant = 800  #800 #  dissipation constant for forces, viscous damping coefficient
        rotational_damping_constant = 1e-4  #  dissipation constant for torques
        gains = np.array([1.0, 1.0, 1.0])

        # pressure transient parameters
        p_min = 0.2
        t_rise = 0.2  # 500 ms / 5 * 2
        t_drop = 0.56  # 1400 ms / 5 * 2

        # numerical parameters
        n_segments = 20                             # number of segments of the rod (n_segments+1 nodes)
        dt = 2.0e-4
        print("dt empirical: %.4e" % (0.01 * module_length / n_segments))
        print("dt actual:    %.4e" % dt)

        # joint parameters
        k_spring = 1e5  # k:  Spring constant of force holding rods together (F = k*x).
        nu_spring = 0  # nu: Energy dissipation of joint nu.
        kt_spring = 1e1  # 5e2  # kt: Rotational stiffness of rod to avoid rods twisting kt.
        nut_spring = 0         # nut: rotational damping coefficient of the joint (default=0)

        robot_sampling_frequency = 10
        teq = 0.1
        save_static = False

        numeric_parameters = {'n_segments': n_segments, 'dt': dt}

        geometric_parameters = {'base_position': start,
                                'base_position_2': start_2,
                                'robot_direction': direction,
                                'normal': normal,
                                'module_length': module_length,
                                'section_radius': section_radius,
                                'delta': delta,
                                'actuator_outer_radius': actuator_outer_radius,
                                'actuator_wall_thickness': actuator_wall_thickness}

        material_parameters = {'weight': weight,
                            'density': density,
                            'young_modulus': young_modulus,
                            'poisson_ratio': poisson_ratio,
                            'translational_damping_constant': translational_damping_constant,
                            'rotational_damping_constant': rotational_damping_constant,
                            'gains': gains
                            }

        valve_parameters = {'p_min': p_min, 't_rise': t_rise, 't_drop': t_drop}
        environment_parameters = {'g': g_vector}

        simulation_parameters = {'robot_sampling_frequency': robot_sampling_frequency,
                                'save_static': save_static,
                                'teq': teq}

        joint_parameters = {'spring_constant': k_spring,
                            'joint_dissipation': nu_spring,
                            'rotational_stiffness': kt_spring,
                            'rotational_damping': nut_spring}
        return numeric_parameters, geometric_parameters, material_parameters, valve_parameters, environment_parameters, simulation_parameters, joint_parameters
    
## Domain 2 Deterministic Model
class ISupportParamD2():
    def __init__(self):
        pass
    def paramaters(self):
        # environment parameters
        g = 9.80513  # gravity in Pisa

        g_vector = np.array([0.0, -g, 0.0])  # (yzx)

        # geometric parameters, spatial reference=(y,z,x)
        start     = np.array([0.0,  0.0, 0.0])       # rod location
        direction = np.array([0.0, -1.0, 0.0])       # rod orientation
        normal    = np.array([0.0,  0.0, -1.0])      # rod normal direction [0,0,-1]

        module_length  = 0.19 # rod length [m]
        section_radius = 0.03 # rod radius (section) [m]
        delta          = 0.02 # distance between section centre and actuator centre [m]
        actuator_outer_radius = 0.00779 # media outer
        actuator_wall_thickness = 0.0014 # wall thickness of 3D printed actuator (1.4 mm)

        start_2 = start + direction * module_length

        # material parameters
        weight         = 0.1704 # (~0.181) total_weight                # manipulator weight          [Kg]       (180 g)
        density        = 1104   # average manipulator density [Kg]/[m]^3
        young_modulus  = 1.646439e+06  ###1.646439e+06--------------> real value
        poisson_ratio  = 0.5
        translational_damping_constant = 8000 #  dissipation constant for forces, viscous damping coefficient
        rotational_damping_constant = 1e-4  #  dissipation constant for torques
        gains = np.array([1.0, 1.0, 1.0])

        # pressure transient parameters
        p_min = 0.2
        t_rise = 0.2  # 500 ms / 5 * 2
        t_drop = 0.56  # 1400 ms / 5 * 2

        # numerical parameters
        n_segments = 20                             # number of segments of the rod (n_segments+1 nodes)
        dt = 2.0e-4
        print("dt empirical: %.4e" % (0.01 * module_length / n_segments))
        print("dt actual:    %.4e" % dt)

        # joint parameters
        k_spring = 1e5  # k:  Spring constant of force holding rods together (F = k*x).
        nu_spring = 0  # nu: Energy dissipation of joint nu.
        kt_spring = 1e1  # 5e2  # kt: Rotational stiffness of rod to avoid rods twisting kt.
        nut_spring = 0         # nut: rotational damping coefficient of the joint (default=0)

        robot_sampling_frequency = 10
        teq = 0.1
        save_static = False

        numeric_parameters = {'n_segments': n_segments, 'dt': dt}

        geometric_parameters = {'base_position': start,
                                'base_position_2': start_2,
                                'robot_direction': direction,
                                'normal': normal,
                                'module_length': module_length,
                                'section_radius': section_radius,
                                'delta': delta,
                                'actuator_outer_radius': actuator_outer_radius,
                                'actuator_wall_thickness': actuator_wall_thickness}

        material_parameters = {'weight': weight,
                            'density': density,
                            'young_modulus': young_modulus,
                            'poisson_ratio': poisson_ratio,
                            'translational_damping_constant': translational_damping_constant,
                            'rotational_damping_constant': rotational_damping_constant,
                            'gains': gains
                            }

        valve_parameters = {'p_min': p_min, 't_rise': t_rise, 't_drop': t_drop}
        environment_parameters = {'g': g_vector}

        simulation_parameters = {'robot_sampling_frequency': robot_sampling_frequency,
                                'save_static': save_static,
                                'teq': teq}

        joint_parameters = {'spring_constant': k_spring,
                            'joint_dissipation': nu_spring,
                            'rotational_stiffness': kt_spring,
                            'rotational_damping': nut_spring}
        return numeric_parameters, geometric_parameters, material_parameters, valve_parameters, environment_parameters, simulation_parameters, joint_parameters
    