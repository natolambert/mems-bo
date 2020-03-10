
def jumper(parameters):
    rho = 2300
    E = 170e9
    T = 550e-6
    lamb = 13.75
    w = parameters["w"]  # T / lamb # min: 4e-5
    g = w
    strain_max = 0.5e-2
    x_max = 5e-3
    Fmax = 15e-3
    Smax = 0.5e-2
    # L range is 500e-6 to 1.9e-2
    # N is range 1 to 6
    N = parameters["N"]
    L = parameters["L"]

    L_mm = L * 1000
    k = 8 * (E * w ** 3 * T) / (N * L ** 3)
    x = x_max
    x_cant = x / (4 * N)
    x_mm = x * 1000
    F_required = k * x
    F_required_mN = F_required * 1000
    strain = (3 / 2) * (x_cant) * w / (L / 4) ** 2
    spring_mass_mg = 2 * N * L * w * T * rho * 1e6
    spring_height = N * (2 * w + g)
    spring_height_mm = spring_height * 1000
    U_stored = 0.5 * k * x ** 2

    # if violates constraints, return -1 (block via outcome constraints)
    const_maxF = (Fmax > k * x)
    const_maxS = (Smax > (3 / 2) * (x_cant) * w / (L / 4) ** 2)
    if const_maxF or const_maxS:
        return -1, 0.0

    # Final results
    frame_mass = ((300e-6 * 11048e-6 * 2) + (300e-6 * 19125e-6 * 2)) * 550e-6 * 2300
    robot_mass = (5507e-6 * 7153e-6 - 2 * 795e-6 * 1678e-6) * 592e-6 * 2300
    total_mass = frame_mass + robot_mass + spring_mass_mg * 1e-6
    total_mass_mg = total_mass * 1e6
    robot_frame_mass_ratio = robot_mass / frame_mass

    jump_height_cm_10mN = ((10e-3) ** 2 / (2 * k)) / (9.8 * total_mass) * 100
    x_mm_10mN = 10e-3 / k * 1e3
    jump_height_cm_12mN = ((12e-3) ** 2 / (2 * k)) / (9.8 * total_mass) * 100
    x_mm_12mN = 12e-3 / k * 1e3
    jump_height_cm_15mN = ((15e-3) ** 2 / (2 * k)) / (9.8 * total_mass) * 100
    x_mm_15mN = 15e-3 / k * 1e3
    x_mm_F_required_best = F_required / k * 1e3
    energy_stored_uJ_F_required_best = 0.5 * k * (F_required / k) ** 2 * 1e6

    return {"Energy": (U_stored, 0.0),
            "Strain": (strain, 0.0),
            "Force": (F_required, 0.0)}
