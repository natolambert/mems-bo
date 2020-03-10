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

    return {"Energy": (U_stored, 0.0),
            "Strain": (strain, 0.0),
            "Force": (F_required, 0.0)}
