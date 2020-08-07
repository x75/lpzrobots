#!/usr/bin/env python
"""empirical predictive information, compare playfulmachines
$ python hk.py -h"""

# FIXME: put the learner / control structure into class to easily load
#        der/martius or reservoir model
# FIXME: control listeners for eta, theta, soft_lim, ...

import time, argparse, sys
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray

if "/home/src/QK/math/neural/dmps" not in sys.path:
    sys.path.insert(0, "/home/src/QK/smp/dmps")

from reservoirs import Reservoir, Reservoir2

from jpype import startJVM, getDefaultJVMPath, JPackage, shutdownJVM, isThreadAttachedToJVM, attachThreadToJVM


# information dynamics toolkit initialization
jarLocation = "../../../../infodynamics-dist/infodynamics.jar"
# print getDefaultJVMPath()
startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

class LPZRosEH(object):
    modes = {"eh_pi_d": 2, "eh_ais_d": 3, "eh_pi_c_l": 4, "eh_var": 5, "eh_ent_d": 6, "eh_pi_c_avg": 7}

    base = 1000
    basehalf = base/2
    
    entCalcClassD = JPackage("infodynamics.measures.discrete").EntropyCalculatorDiscrete
    entCalc = entCalcClassD(base)

    piCalcClass = JPackage("infodynamics.measures.discrete").PredictiveInformationCalculatorDiscrete
    # print piCalcClass
    piCalc10 = piCalcClass(base,1)

    aisCalcClassD = JPackage("infodynamics.measures.discrete").ActiveInformationCalculatorDiscrete
    aisCalc = aisCalcClassD(base,1)
        
    piCalcClassC = JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel
    piCalc = piCalcClassC();
    piCalc.setProperty("NORMALISE", "true"); # Normalise the individual variables
    
    def __init__(self, mode="hs"):
        self.init = False
        self.name = "lpzros"
        self.mode = LPZRosEH.modes[mode]
        self.cnt = 0
        ############################################################
        # ros stuff
        rospy.init_node(self.name)
        # pub=rospy.Publisher("/motors", Float64MultiArray, queue_size=1)
        self.pub_motors   = rospy.Publisher("/motors", Float64MultiArray)
        self.pub_res_r    = rospy.Publisher("/reservoir/r", Float64MultiArray)
        self.pub_res_w    = rospy.Publisher("/reservoir/w", Float64MultiArray)
        self.pub_res_perf = rospy.Publisher("/reservoir/perf", Float64MultiArray)
        self.pub_res_perf_lp = rospy.Publisher("/reservoir/perf_lp", Float64MultiArray)
        self.pub_res_mdltr = rospy.Publisher("/reservoir/mdltr", Float64MultiArray)
        self.sub_sensor   = rospy.Subscriber("/sensors", Float64MultiArray, self.cb_sensors)
        # pub=rospy.Publisher("/chatter", Float64MultiArray)
        self.msg          = Float64MultiArray()
        self.msg_res_r    = Float64MultiArray()
        self.msg_res_w    = Float64MultiArray()
        self.msg_res_perf = Float64MultiArray()
        self.msg_res_perf_lp = Float64MultiArray()
        self.msg_res_mdltr = Float64MultiArray()

        # controller
        self.N = 200
        self.p = 0.1
        # self.g = 1.5
        # self.g = 1.2
        self.g = 0.999
        # self.g = 0.001
        # self.g = 0.
        self.alpha = 1.0

        self.wf_amp = 0.0
        # self.wf_amp = 0.005
        # self.wi_amp = 0.01
        # self.wi_amp = 0.1
        # self.wi_amp = 1.0 # barrel
        self.wi_amp = 5.0

        self.idim = 2
        self.odim = 2

        # simulation time / experiment duration
        # self.nsecs = 500 #1440*10
        # self.nsecs = 250
        # self.nsecs = 100
        self.nsecs = 50 #1440*10
        self.dt = 0.005
        self.dt = 0.01
        self.learn_every = 1
        self.test_rate = 0.95
        self.washout_rate = 0.05
        self.simtime = np.arange(0, self.nsecs, self.dt)
        self.simtime_len = len(self.simtime)
        self.simtime2 = np.arange(1*self.nsecs, 2*self.nsecs, self.dt)

        # self.x0 = 0.5*np.random.normal(size=(self.N,1))
        self.x0 = 0.5 * np.random.normal(size=(self.N, 1))
        self.z0 = 0.5 * np.random.normal(size=(1, self.odim))

        # EH stuff
        self.z_t = np.zeros((2,self.simtime_len))
        self.zn_t = np.zeros((2,self.simtime_len)) # noisy readout
        # self.zp_t = np.zeros((2,self.simtime_len))
        self.wo_len_t = np.zeros((2,self.simtime_len))
        self.dw_len_t = np.zeros((2,self.simtime_len))
        self.perf = np.zeros((1,self.odim))
        self.perf_t = np.zeros((2,self.simtime_len))
        self.mdltr = np.zeros((1,2))
        self.mdltr_t = np.zeros((2,self.simtime_len))
        self.r_t = np.zeros(shape=(self.N, self.simtime_len))
        self.zn_lp = np.zeros((1,2))
        self.perf_lp = np.zeros((1,2))
        self.perf_lp_t = np.zeros((2,self.simtime_len))
        # self.coeff_a = 0.2
        # self.coeff_a = 0.1
        self.coeff_a = 0.05
        # self.coeff_a = 0.03
        # self.coeff_a = 0.001
        # self.coeff_a = 0.0001
        # self.eta_init = 0.0001
        # self.eta_init = 0.001 # limit energy in perf
        self.eta_init = 0.0025 # limit energy in perf
        # self.eta_init = 0.01 # limit energy in perf
        self.T = 200000.
        # self.T = 2000.

        # exploration noise
        # self.theta = 1.0
        # self.theta = 1/4.
        self.theta = 0.1
        # self.state noise
        self.theta_state = 0.1
        # self.theta_state = 1e-3

        # leaky integration time constant
        # self.tau
        # self.tau = 0.001
        # self.tau = 0.005
        self.tau = 0.2
        # self.tau = 0.9
        # self.tau = 0.99
        # self.tau = 1.

        ############################################################
        # alternative controller network / reservoir from lib
        # FIXME: use config file
        # FIXME: use input coupling matrix for non-homogenous input scaling
        self.res = Reservoir2(N = self.N, p = self.p, g = self.g, alpha = 1.0, tau = self.tau,
                         input_num = self.idim, output_num = self.odim, input_scale = self.wi_amp,
                         feedback_scale = self.wf_amp, bias_scale = 0., eta_init = self.eta_init,
                         sparse=True)
        self.res.theta = self.theta
        self.res.theta_state = self.theta_state
        self.res.coeff_a = self.coeff_a

        self.res.x = self.x0
        self.res.r = np.tanh(self.res.x)
        self.res.z = self.z0
        self.res.zn = np.atleast_2d(self.z0).T

        # print res.x, res.r, res.z, res.zn

        print('   N: ', self.N)
        print('   g: ', self.g)
        print('   p: ', self.p)
        print('   nsecs: ', self.nsecs)
        print('   learn_every: ', self.learn_every)
        print('   theta: ', self.theta)

        # set point for goal based learning
        self.sp = 0.4

        # explicit memory
        # self.piwin = 100
        # self.piwin = 200
        # self.piwin = 500
        # self.piwin = 1000
        self.piwin = 2000

        self.x_t = np.zeros((self.idim, self.piwin))
        self.z_t = np.zeros((self.odim, self.piwin))

        # self.wgt_lim = 0.5 # barrel
        self.wgt_lim = 0.1
        self.wgt_lim_inv = 1/self.wgt_lim
        
        self.init = True
        print("init done")

    def soft_bound(self):
        # FIXME: for now its actually hard bounded
        # FIXME: modulate self.eta_init for effective soft bounds
        # FIXME: decouple the readouts / investigate coupled vs. uncoupled
        # 1 determine norm
        wo_norm = np.linalg.norm(self.res.wo, 2)
        print("|wo| =", wo_norm)
        # 2 scale weights down relatively to some setpoint norm
        if wo_norm > self.wgt_lim:
            self.res.wo /= (wo_norm * self.wgt_lim_inv)
            # 3 slight randomization / single weight flips?
            for ro_idx in range(self.odim):
                if np.random.uniform(0., 1.) > 0.95:
                    numchoice = np.random.randint(0, 5)
                    selidx = np.random.choice(self.N, numchoice, replace=False)
                    print("randomize weights", selidx)
                    # self.res.wo[selidx, ro_idx] += np.random.normal(self.res.wo[selidx, ro_idx], 0.001)
                    # reduce weights only
                    self.res.wo[selidx, ro_idx] -= np.random.exponential(0.001, numchoice)
                    
    def cb_sensors(self, msg):
        """lpz sensors callback: receive sensor values, sos algorithm attached"""
        if not self.init: return
        # self.msg.data = []
        self.x_t = np.roll(self.x_t, -1, axis=1) # push back past
        # self.z = np.roll(self.y, 1, axis=1) # push back past

        # update input with new sensor data
        u = np.reshape(np.asarray(msg.data), (self.idim, 1))
        self.x_t[:,-1] = u.reshape((self.idim,))
        # compute network output
        self.res.execute(u)
        # print self.res.z

        # learning
        dw = 0
        # if self.cnt > 0: # self.piwin:
        if self.cnt > self.piwin:
            for sysdim in range(self.idim):
                attachThreadToJVM()
                # x_tmp = self.x_t[sysdim,:]
                # x_tmp = (((self.x_t[sysdim,:] / (2*np.pi)) + 0.5) * 999).astype(int)
                # discrete ENT / PI / AIS / ...
                # FIXME: use digitize and determine bin boundaries from min/max
                x_tmp = (((self.x_t[sysdim,:] / 3.) + 0.5) * 999).astype(int)
                # print "x_tmp", sysdim, x_tmp
                # x_tmp = x_tmp - np.min(x_tmp)
                # x_tmp = ((x_tmp / np.max(x_tmp)) * (LPZRosEH.base-1)).astype(int)
                # print "x_tmp", x_tmp
                # # print "jvm", isThreadAttachedToJVM()
                # pis = LPZRosEH.piCalc10.computeLocal(x_tmp)
                if self.mode == LPZRosEH.modes["eh_ent_d"]: # EH learning, discrete PI
                    # plain entropy
                    pis = LPZRosEH.entCalc.computeLocal(x_tmp)
                    self.perf[0,sysdim] = list(pis)[-1] * -1
                elif self.mode == LPZRosEH.modes["eh_pi_d"]: # EH learning, discrete PI
                    # predictive information
                    # pis = LPZRosEH.piCalc10.computeLocal(x_tmp)
                    # self.perf[0,sysdim] = list(pis)[-1]
                    pis = LPZRosEH.piCalc10.computeAverageLocal(x_tmp)
                    self.perf[0,sysdim] = pis
                elif self.mode == LPZRosEH.modes["eh_ais_d"]: # EH learning, discrete PI
                    # pis = LPZRosEH.aisCalc.computeLocal(x_tmp)
                    # self.perf[0,sysdim] = list(pis)[-1]
                    pis = LPZRosEH.aisCalc.computeAverageLocal(x_tmp)
                    self.perf[0,sysdim] = pis
                elif self.mode == LPZRosEH.modes["eh_pi_c_l"]: # EH learning, discrete PI
                    # local continuous predictive information
                    LPZRosEH.piCalc.initialise(1, 1, 0.5); # Use history length 1 (Schreiber k=1),
                    x_src = np.atleast_2d(self.x_t[sysdim,0:-1]).T
                    x_dst = np.atleast_2d(self.x_t[sysdim,1:]).T
                    LPZRosEH.piCalc.setObservations(x_src, x_dst)
                    pis = LPZRosEH.piCalc.computeLocalOfPreviousObservations()
                    self.perf[0,sysdim] = list(pis)[-1]
                elif self.mode == LPZRosEH.modes["eh_pi_c_avg"]: # EH learning, discrete PI
                    # average continuous predictive information
                    LPZRosEH.piCalc.initialise(1, 1, 0.5); # Use history length 1 (Schreiber k=1),
                    x_src = np.atleast_2d(self.x_t[sysdim,0:-1]).T
                    x_dst = np.atleast_2d(self.x_t[sysdim,1:]).T
                    LPZRosEH.piCalc.setObservations(x_src, x_dst)
                    self.perf[0,sysdim] = LPZRosEH.piCalc.computeAverageLocalOfObservations()
                elif self.mode == LPZRosEH.modes["eh_var"]: # EH learning, discrete PI
                    # variance
                    self.perf[0,sysdim] = np.var(self.x_t[sysdim,:])
                else:
                    self.perf[0,sysdim] = 0.
                    # print "pis", pis
            print("perf", self.perf)

            # recent performance
            self.perf_lp = self.perf_lp * (1 - self.coeff_a) + self.perf * self.coeff_a
            
            ############################################################
            # learning
            # FIXME: put that into res / model member function
            for ro_idx in range(self.odim):
                # if gaussian / acc based
                # if perf[0, ro_idx] > (perf_lp[0, ro_idx] + 0.1):
                # for information based
                # FIXME: consider single perf / modulator for all readouts
                if self.perf[0, ro_idx] > self.perf_lp[0, ro_idx]:
                    self.mdltr[0, ro_idx] = 1.
                else:
                    self.mdltr[0, ro_idx] = 0.
            eta = self.eta_init / (1 + (self.cnt/self.T))
            # eta = eta_init
            # dw = eta * (zn_t[0, ti - 20] - zn_lp) * mdltr * r
            # dw = eta * (zn.T - zn_lp) * mdltr * r
            # print "2D dbg" zn.shape, zn_lp.shape
            if True: # self.cnt < self.test_rate * self.simtime_len:
                # dw = eta * (zn.T - zn_lp) * mdltr * r
                # wo += dw
                dw = eta * (self.res.zn.T - self.res.zn_lp.T) * self.mdltr * self.res.r
                # print dw
                self.res.wo += dw
                # FIXME: apply soft bounding on weights or weight decay
                self.soft_bound()
            else:
                dw = np.zeros(self.res.r.shape)
                self.mdltr[0,:] = 0.

                # if np.abs(ip2d.x[ti,0]) > 10. or np.abs(ip2d.x[ti,1]) > 10.:
                #     sys.exit()

        # # check this
        # bins = np.arange(-1, 1.1, 0.1)
        # x_tmp = np.digitize(x[:,0], bins)

        # base = np.max(x_tmp)+1 # 1000
        # basehalf = base/2
        # piCalc10 = piCalcClassD(base,3)
        # aisCalc10 = aisCalcClassD(base,5)

        # pi = list(piCalc10.computeLocal(x_tmp))
        # ais = list(aisCalc10.computeLocal(x_tmp))
                    
        self.msg.data = self.res.zn.flatten().tolist()
        # print self.msg.data
        # print("sending msg", msg)
        self.pub_motors.publish(self.msg)
        self.msg_res_r.data = self.res.r.flatten().tolist()
        self.pub_res_r.publish(self.msg_res_r)
        self.msg_res_w.data = np.linalg.norm(self.res.wo, 2, axis=0)
        self.pub_res_w.publish(self.msg_res_w)
        self.msg_res_perf.data = self.perf.flatten().tolist()
        self.pub_res_perf.publish(self.msg_res_perf)
        self.msg_res_perf_lp.data = self.perf_lp.flatten().tolist()
        self.pub_res_perf_lp.publish(self.msg_res_perf_lp)
        self.msg_res_mdltr.data = self.mdltr.flatten().tolist()
        self.pub_res_mdltr.publish(self.msg_res_mdltr)
        # time.sleep(0.1)
        # if self.cnt > 20:
        #     rospy.signal_shutdown("stop")
        #     sys.exit(0)
        self.cnt += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lpzrobots ROS controller: test empiricial predictive information")
    parser.add_argument("-m", "--mode", type=str, help="select mode: " + str(LPZRosEH.modes),
                        default="eh_pi_d")
    # parser.add_argument("-m", "--mode", type=int, help="select mode: ")
    args = parser.parse_args()


    print("jvm", isThreadAttachedToJVM())
    # sanity check
    if not args.mode in LPZRosEH.modes:
        print("invalid mode string, use one of " + str(LPZRosEH.modes))
        sys.exit(0)
    
    lpzros = LPZRosEH(args.mode)
    rospy.spin()
    # while not rospy.shutdown():
