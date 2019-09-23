
## DEFINE SENSOR FUSION ALGORITHM ##

### SET UP KALMAN FILTER ###
# initialization of the state vector x. It contains the the 3 coordinates of a point
x0 = np.array([0,0,0])
x_prev_posterior = x0
x_prev_posterior_vel = np.zeros(6)
# state transition matrix A. Here, the simple assumption is that the location of the
# point doesn't change.
A = np.identity(3)
# A_vel is updated every time with the delta t's, since we don't assume constant delta t's

# define the process covariance Q. That's the covariance of the error of the
# pure state transition model. Our state transition model is expected to have
# a relatively high error variance <=> a lot of noise, since it is often not true
# that the point is not moving as our state transition model claims.
Q = np.array([[0.1**2, 0., 0.],
              [0., 1**2, 0.],
              [0., 0., 0.1**2]])
Q_vel = np.zeros((6, 6), float)
Q_vel[0:3, 0:3] = Q
Q_vel[3:6, 3:6] = np.array([[0.05**2, 0., 0.],
                            [0., 0.05**2, 0.],
                            [0., 0., 0.05**2]])

#This Q lead to the kalman gain sabilizing at I*0.83. A lower kalman gain
#could be better due to the bad aruco localization. Parameter tuning:
Q = Q*0.3
# is leads to a Kalman gain of I*0.65, which seems to be a bit better.

# initialize the covariance matrix of the posterior state estimation error x_k_true - x_k_posterior
# We use some reasonable values. Here we use Q/2, pretending the posterior state
# estimation has half the noise of the prior state estimation which comes from the
# state transition model. P_k will be updated frequently anyway, so the initialization
# is not too important.
P0 = np.array([[0.1**2, 0., 0.],
              [0., 1**2, 0.],
              [0., 0., 0.1**2]])/2
P_prev_posterior = P0
P_prev_posterior_vel = np.identity(6) * 0.01

# R is the covariance matrix of the measurement error x_k_true - z_k
# First, have a look of xlim, ylim, zlim of the plot furhter below
# we expect x and z values roughly between -0.3 and 0.3 meters so assuming
# standard deviation (sigma) of 0.05 for the x and z measurement is resonable.
# This means it is expected that in 95% of the measurements, their errors are
# between +-1.96*sigma = +-1.96*0.1, leading to an 95%-confidence-interval length of 0.196 or roughly 0.2 meters.
# std of 0.05 means a var of 0.05^2 = 0.0025
# we expect y values roughly between 0 and 4 meters, so assuming a std of
# 0.5 meters for the y measurement is reasonable, which means a
# 95%-confidence-inteveral length of the y measurement error to be 1.96 or roughly 2 meters
# std of 0.5 meters means a var of sqrt(0.5)
R = np.array([[0.05**2, 0., 0.],
              [0., 0.5**2, 0.],
              [0., 0., 0.05**2]]) #should be deprecated later

# steady point sensors
R_wc = np.array([[0.05**2, 0., 0.],
              [0., 0.5**2, 0.],
              [0., 0., 0.05**2]])
R_kc = np.array([[0.06**2, 0., 0.],
              [0., 0.6**2, 0.],
              [0., 0., 0.06**2]])
R_kd = np.array([0.08**2])
R_sim = np.array([0.08**2])

# point sensors extended by indirect velocity measurement
R_wc_vel = np.zeros((6, 6), float)
R_wc_vel[0:3, 0:3] = R_wc
R_wc_vel[3:6, 3:6] = np.array([[0.005**2, 0., 0.],
                            [0., 0.005**2, 0.],
                            [0., 0., 0.005**2]])

R_kc_vel = np.zeros((6, 6), float)
R_kc_vel[0:3, 0:3] = R_kc
R_kc_vel[3:6, 3:6] = np.array([[0.005**2, 0., 0.],
                            [0., 0.005**2, 0.],
                            [0., 0., 0.005**2]])

R_kd_vel = np.diag([R_kd[0], 0.005**2])

R_sim_vel = np.diag([R_sim[0], 0.005**2])


# Steady point linear kalman update
def kalman_update_steady(z_k):

    global x_prev_posterior
    global P_prev_posterior

    ### TIME UPDATE ###
    x_k_prior = np.matmul(A, x_prev_posterior)
    P_k_prior = P_prev_posterior + Q


    ### MEASURMENT UPDATE ###
    K_k = np.matmul(P_k_prior, np.linalg.inv(P_k_prior + R))
    print(K_k)

    x_k_posterior = x_k_prior + np.matmul(K_k, (z_k - x_k_prior))
    P_k_posterior = np.matmul((np.identity(3) - K_k), P_k_prior)
    print(P_k_posterior)

    # The current _k becomes _prev for the next time step, therefore
    # update the global variables
    x_prev_posterior = x_k_posterior
    P_prev_posterior = P_k_posterior

    return(x_k_posterior)

# Steady point linear kalman update that supports multiple sensor input
def kalman_update_steady_multiple(z_k, H, R):

    global x_prev_posterior
    global P_prev_posterior

    ### TIME UPDATE ###
    x_k_prior = np.matmul(A, x_prev_posterior)
    P_k_prior = P_prev_posterior + Q


    ### MEASURMENT UPDATE ###
    K_k = np.matmul(np.matmul(P_k_prior, H.transpose()), np.linalg.inv(H @ P_k_prior @ H.transpose() + R))
    print("Kalman Gain:")
    print(K_k)

    x_k_posterior = x_k_prior + np.matmul(K_k, (z_k - np.matmul(H, x_k_prior)))
    P_k_posterior = np.matmul((np.identity(3) - np.matmul(K_k, H)), P_k_prior)
    print(P_k_posterior)

    # The current _k becomes _prev for the next time step, therefore
    # update the global variables
    x_prev_posterior = x_k_posterior
    P_prev_posterior = P_k_posterior

    print("kalman estimatoin:")
    print(x_k_posterior)

    return(x_k_posterior)

# constant velocity linear kalman update
def kalman_update_velocity(z_k, H, R, A):
    global x_prev_posterior_vel
    global P_prev_posterior_vel

    ### TIME UPDATE ###
    x_k_prior = np.matmul(A, x_prev_posterior_vel)
    P_k_prior = A @ P_prev_posterior_vel @ A.transpose() + Q_vel


    ### MEASURMENT UPDATE ###
    K_k = np.matmul(np.matmul(P_k_prior, H.transpose()), np.linalg.inv(H @ P_k_prior @ H.transpose() + R))
    print("Kalman Gain:")
    print(K_k)

    x_k_posterior = x_k_prior + np.matmul(K_k, (z_k - np.matmul(H, x_k_prior)))
    P_k_posterior = np.matmul((np.identity(6) - np.matmul(K_k, H)), P_k_prior)
    print(P_k_posterior)

    # The current _k becomes _prev for the next time step, therefore
    # update the global variables
    x_prev_posterior_vel = x_k_posterior
    P_prev_posterior_vel = P_k_posterior

    print("kalman estimatoin:")
    print(x_k_posterior)

    return(x_k_posterior)

#kalman_estimation expects a list of listes of localizations and identifiers of different sensors
# e.g. update from kinect cam and kinect depth sensor:
# e.g. sensor_readings, sensor_names = [kc_xyzt, kd_zt], ['kc', 'kd']
# old: [[kc_xyzt, 'kc'], ['kc, 'kd']]
# For testing:
sensor_readings = [np.array([0.5, 0.5, 0.5]), np.array([0.4])]
reading_times = [123456, 123456]
sensor_names = ['kc', 'kd']


# Storing values of previous invocations of kalman_estmiation, necessary to
# calculate differences dx, dy, dz, dt and therefore velocities vx, vy, vz
wc_xyz_prev = kc_xyz_prev = np.array([0,0,0])
kd_z_prev = sim_z_prev = np.array([0])
wc_t_prev = kc_t_prev = kd_t_prev = sim_t_prev = time.time()




def filter(dtime, coords):

	# TODO copy kalman_estimation after matching function definition to call

	return coords[0]

