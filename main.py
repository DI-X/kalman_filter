import random
import csv
import numpy as np
from numpy import random as nr

class test_klman:
    def __init__(self):
        self.dt = 0.01

        self.F=np.array([[1, self.dt],
                    [0, 1]])

        self.B=np.array([self.dt*self.dt/2, self.dt])

        self.H = np.array([[1,0],
                          [0,1]])

        # self.H = np.array([1,0])

        self.Q=np.array([[1,0],
                         [0,1]])

        self.R=np.array([[1,0],
                         [0,1]])

        self.X_pri = np.zeros(2)
        self.X_pos = np.zeros(2)

        self.X_act = np.array([0,0.0])

        self.P_pri = np.zeros([2,2])
        self.P_pos = np.zeros([2,2])

        self.counter = 0
        self.t = 0
        self.data2log = list()

    def estmate(self, z, acc):
        self.X_pri = self.F @ self.X_pos + self.B * acc
        self.P_pri = self.F @ self.P_pos @ self.F.T + self.Q
        K_gain = self.P_pri @ self.H.T @ np.linalg.inv(self.H @ self.P_pri @ self.H.T + self.R)

        self.X_pos = self.X_pri + K_gain @ (z - self.H @ self.X_pri)
        self.P_pos = self.P_pri - K_gain @ self.H @ self.P_pri

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
        noise_gus = np.array([nr.normal(loc=0, scale=1),nr.normal(loc=0, scale=1)])

        z = self.X_act #+ noise_gus

        est = self.estmate(z,acc)

        data = list()
        data.append(str(self.t))
        data.append(str(acc))
        data.append(str(self.X_act[0]))
        data.append(str(self.X_act[1]))
        data.append(str(z[0]))
        data.append(str(z[1]))
        data.append(str(est[0]))
        data.append(str(est[1]))
        data.append(str(noise_ran[1]))
        data.append(str(noise_gus[1]))

        self.data2log.append(data)

    def logData(self):
        header = ['/time', '/acc',
                  '/act/pos','/act/vel',
                  '/mes/pos','/mes/vel',
                  '/est/pos','/est/vel',
                  '/noise/random', '/noise/gussian']

        with open('log_data.csv', 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(self.data2log)):
                writer.writerow(self.data2log[i])

            print("save data!")


obj = test_klman()

while obj.t<15:
    obj.run()
obj.logData()





