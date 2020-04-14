def gca(parameters):
    # Conversion of Craig Schindler's Matlab code: Force_vs_Velocity_test_structures_theoretical.mat
    # solves the ode for the force vs velocity of an ideal gap closer
    # imports
    import numpy as np
    from scipy import integrate
    import matplotlib.pyplot as plt

    # Defining Constants
    eps0 = 8.85e-12
    E = 170e9
    t = 40e-6
    gf = 4.833e-6
    gs = 1e-6
    x_GCA = 3.833e-6
    V = parameters["V"] #V 
    changeFactor = 100.0

    Lol = 76e-6
    A = t * Lol
    C = eps0 * A / gf
    Nfing = 96.0
    Ctot = Nfing * C
    Fmin_mN = (1 / 2) * V ** 2 * Ctot / gf * 1e3
    L = parameters["L"] #um
    N = 16.0

    Farr = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
    Farr = np.multiply(Farr, 1e-6)
    karr = np.divide(Farr, changeFactor * x_GCA)

    warr_um = np.divide(karr, (E * t / (N * L ** 3)) ** (1 / 3) * 1e6)
    warr_um_drawn = np.add(warr_um, 1)
    strainarr_percent = 3 * np.multiply((np.divide(warr_um, 1e6)), changeFactor * x_GCA / (2 * N) / (2 * (L / 2) ** 2) * 100)
        # k = E * w**3 * t / (N * L**3)

    int_time = np.array([0, 0.5])


    def pull_in(t, x):
        eps0 = 8.85e-12
        Lol = 76e-6
        t_SOI = 40e-6
        gf = 4.833e-6
        gb = 7.75e-6
        k_support = 50
        N_fing = 96
        Fload_pullin = x[2]
        V = x[3]

        Fes = N_fing * (1 / 2) * V ** 2 * eps0 * t_SOI * Lol * (1 / (gf - x[0]) ** 2 - 1 / (gb + x[0]) ** 2)
        Fd = x[1] * N_fing * 1.85e-5 * Lol * t_SOI ** 3 / (gf - x[0]) ** 3
        Fk = k_support * x[0]
        m = ((1350e-6 * 20e-6) + 96 * (76e-6 * 5e-6) - 56 * (8e-6 * 8e-6)) * 40e-6 * 2300
        m_spring = (600e-6 * 8e-6 * 40e-6 * 2300 * 16) * (1 / 3)  # spring effective mass, has a mass of 1/3 its actual mass
        m_spring = 0  # remove this to actually count the spring mass
        m = m + (m_spring / 3)  # shuttle mass + effective spring mass
        dxdt = [[x[1]], [(Fes - Fd - Fk - Fload_pullin) / m], [0], [0]]
        return dxdt


    def pulled_in(x, t):
        value = (x[0] >= 3.833e-6)
        isterminal = 1  # Stop the integration
        direction = 0
        return [value, isterminal, direction]

    from scipy.integrate import odeint

    for V in range(40, 110, 10):
        sol_vec = []  # initialize sol_vec
        eps0 = 8.85e-12
        Lol = 76e-6
        t_SOI = 40e-6
        gf = 4.833e-6
        gb = 7.75e-6
        k_support = 50
        N_fing = 96
        Fmin = N_fing * (1 / 2) * V ** 2 * eps0 * t_SOI * Lol * (1 / (gf) ** 2 - 1 / (gb) ** 2)

        for Fload in range(0, 402, 2):
            Fload = Fload * 1e-6
            if Fload > 0.995 * Fmin:
                break
            # x0 = np.dtype('Float64')
            x0 = np.array([0, 0, Fload, V])
            # opt = odeset('Events', @pulled_in)
            # [t,x] = ode45(@pull_in,int_time,x0,opt)

            # def vdp1(t, y):
            #     return np.array([y[1], (1 - y[0] ** 2) * y[1] - y[0]])

            ## code adapted from https://stackoverflow.com/questions/48428140/imitate-ode45-function-from-matlab-in-python
            t0, t1 = 0, 20  # start and end
            t = np.linspace(t0, t1, 100)  # the points of evaluation of solution
            # y0 = [2, 0]  # initial value
            x = np.zeros((len(t), len(x0)))  # array for solution
            for i in range(len(x0)):
                    x[0] = x0[0]
            r = integrate.ode(pull_in).set_integrator("dopri5")  # choice of method
            r.set_initial_value(x0, t0)  # initial values
            for i in range(1, t.size):
                x[i] = r.integrate(t[i])  # get one more value, add it to the array
                if not r.successful():
                    raise RuntimeError("Could not integrate")

            # sol_vec = [[sol_vec], [1e6*Fload, 1e6*t(len(t)-1), (gf-gs)/t(len(t)-1), 1e6*Fload*(gf-gs)/t(len(t)-1)]]
    plt.plot(t, x)
    plt.show()