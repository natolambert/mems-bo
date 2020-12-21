def jumper(parameters):
    rho = 2300
    E = 170e9
    T = 550e-6
    lamb = 13.75
    w = parameters["w"]
    x_max = 5e-3
    N = parameters["N"]
    L = parameters["L"]

    k = 8 * (E * w ** 3 * T) / (N * L ** 3)
    x = x_max
    x_cant = x / (4 * N)
    F_required = k * x
    strain = (3 / 2) * (x_cant) * w / (L / 4) ** 2
    U_stored = 0.5 * k * x ** 2

    spring_mass_mg = 2 * N * L * w * T * rho * 1e6
    frame_mass = ((300e-6 * 11048e-6 * 2) + (300e-6 * 19125e-6 * 2)) * 550e-6 * 2300
    robot_mass = (5507e-6 * 7153e-6 - 2 * 795e-6 * 1678e-6) * 592e-6 * 2300
    total_mass = frame_mass + robot_mass + spring_mass_mg * 1e-6
    total_mass_mg = total_mass * 1e6
    robot_frame_mass_ratio = robot_mass / frame_mass


    return {"Energy_(uJ)": (U_stored/ total_mass_mg, 0.0),
            "Strain": (strain, 0.0),
            "Force_(N)": (F_required, 0.0)}
