import random
import csv
import numpy as np
from numpy import random as nr

class test_klman:
    def __init__(self):
        self.dt = 0.01

        self.F=np.array([[1, 0, 0, self.dt, 0, 0],
                         [0, 1, 0, 0, self.dt, 0],
                         [0, 0, 1, 0, 0, self.dt],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])

        self.B=np.array([self.dt*self.dt/2, 0, 0, self.dt, 0, 0])

        # self.H = np.array([[1,0],
        #                   [0,1]])

        self.H = np.array([[-1, 1, 0, 0, 0, 0],
                           [-1, 0, 1, 0, 0, 0],
                           [0, 0, 0, -1, 1, 0],
                           [0, 0, 0, -1, 0, 1]])

        self.Q=np.eye(6)

        self.R=np.eye(4) # measurement noise

        self.X_pri = np.zeros(6)
        self.X_pos = np.zeros(6)

        self.X_act = np.zeros(6)
        self.X_act[0] = 1
        self.X_act[3] = 0.5

        self.P_pri = np.zeros([6,6])
        self.P_pos = np.zeros([6,6])

        self.counter = 0
        self.t = 0
        self.data2log = list()

        self.flag = np.zeros(2)
        self.small_cov = 0.01
        self.large_cov = 30000
        self.init_pos = np.zeros(2)


    def estmate(self, z, acc):
        self.X_pri = self.F @ self.X_pos + self.B * acc
        self.P_pri = self.F @ self.P_pos @ self.F.T + self.Q
        K_gain = self.P_pri @ self.H.T @ np.linalg.inv(self.H @ self.P_pri @ self.H.T + self.R)

        self.X_pos = self.X_pri + K_gain.dot(z - self.H.dot( self.X_pri))
        self.P_pos = self.P_pri - K_gain.dot(self.H.dot(self.P_pri))

        return self.X_pos

    def updateCounter(self):
        self.counter +=1
        self.t = self.dt* self.counter

    def run(self):

        self.updateCounter()
        mag = 1
        acc = mag * np.sin(2*np.pi*self.t)
        self.X_act = self.F @ self.X_act + self.B * acc

        noise_ran = np.array([random.random(), random.random()])
        noise_gus = np.array([nr.normal(loc=0, scale=1),nr.normal(loc=0, scale=1),nr.normal(loc=0, scale=1),nr.normal(loc=0, scale=1)])

        z = self.H @ self.X_act #+ noise_gus
        if int(self.t%2) == 0:
            self.flag[1] = 0
            if self.flag[0] == 0:
                self.init_pos[0] = self.X_act[0].copy()
                self.flag[0] = 1
            z[0] = self.init_pos[1] - self.X_act[0]
            z[1] = noise_ran[0]
            z[2] = -self.X_act[3]
            z[3] = noise_ran[1]

            self.R[0, 0] = self.small_cov
            self.R[1, 1] = self.large_cov
            self.R[2, 2] = self.small_cov
            self.R[3, 3] = self.large_cov

            self.Q[1,1] = self.small_cov
            self.Q[2, 2] = self.large_cov
            self.Q[4,4] = self.small_cov
            self.Q[5, 5] = self.large_cov
        else:

            self.flag[0] = 0
            if self.flag[1] == 0:
                self.init_pos[1] = self.X_act[0].copy()
                self.flag[1] = 1

            z[0] = noise_ran[0]
            z[1] =  self.init_pos[1] - self.X_act[0]
            z[2] = noise_ran[1]
            z[3] = -self.X_act[3]

            self.R[0, 0] = self.large_cov
            self.R[1, 1] = self.small_cov
            self.R[2, 2] = self.large_cov
            self.R[3, 3] = self.small_cov

            self.Q[1, 1] = self.large_cov
            self.Q[2, 2] = self.small_cov
            self.Q[4, 4] = self.large_cov
            self.Q[5, 5] = self.small_cov


        est = self.estmate(z, acc)
        z_est = self.H @ est + noise_gus

        data = list()
        data.append(str(self.t))
        data.append(str(acc))
        data.append(str(self.X_act[0]))
        data.append(str(self.X_act[1]))
        data.append(str(self.X_act[2]))
        data.append(str(self.X_act[3]))
        data.append(str(self.X_act[4]))
        data.append(str(self.X_act[5]))
        data.append(str(z[0]))
        data.append(str(z[1]))
        data.append(str(z[2]))
        data.append(str(z[3]))
        data.append(str(z_est[0]))
        data.append(str(z_est[1]))
        data.append(str(z_est[2]))
        data.append(str(z_est[3]))
        data.append(str(est[0]))
        data.append(str(est[1]))
        data.append(str(est[2]))
        data.append(str(est[3]))
        data.append(str(est[4]))
        data.append(str(est[5]))
        data.append(str(noise_ran[1]))
        data.append(str(noise_gus[1]))

        self.data2log.append(data)

    def logData(self):
        header = ['/time', '/acc',
                  '/act/pos_com','/act/pos_1','/act/pos_2', '/act/vel_com','/act/vel_1','/act/vel_2',
                  '/mes/p1_pcom','/mes/p2_pcom','/mes/v1_vcom', '/mes/v2_vcom',
                  '/mes_est/p1_pcom', '/mes_est/p2_pcom', '/mes_est/v1_vcom', '/mes_est/v2_vcom',
                  '/est/pos_com','/est/pos_1','/est/pos_2', '/est/vel_com','/est/vel_1','/est/vel_2',
                  '/noise/random', '/noise/gussian']

        with open('log_data.csv', 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(self.data2log)):
                writer.writerow(self.data2log[i])

            print("save data!")


obj = test_klman()

while obj.t<55:
    obj.run()
obj.logData()





