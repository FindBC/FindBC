#!/usr/bin/env python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import fsolve
from scipy.optimize import leastsq
from scipy.optimize import minimize
import getopt
import sys
import scipy.optimize

# conditional print
def print_debug(m):
    if Info: print m

# Computation of equivalent stress
def equivalentStress(Stress):
    spherical_part = 1./3.*np.sum(Stress)
    return sqrt((Stress[0]-spherical_part)**2+(Stress[1]-spherical_part)**2+(Stress[2]-spherical_part)**2)*sqrt(3./2.)

# Computation of equivalent Strain
def equivalentStrain(e):
    return sqrt((e[0]**2+e[1]**2+e[2]**2))*sqrt(2./3.)

# Computation of the pressure
def computePressure(stress):
    return -1./3.*np.sum(stress)

# Computation of the Triaxiality
def triaxiality(stress):
    s = stress-1./3.*np.sum(stress)*np.ones_like(stress)
    return -computePressure(stress)/equivalentStress(s)

# Computation of Lode's angle
def lodeAngle(s):
#    print s
    s_sorted = np.sort(s)[::-1]
#    print s_sorted
#    s_sorted = s
    return (2*s_sorted[1]-s_sorted[2]-s_sorted[0])/(s_sorted[0]-s_sorted[2])

# Computation of deviatoric stress (Norton-Hoff)
def deviatoricStressComponent(e, params, comp=-1):
    A = params['A']
    m = params['m']
    s = np.array([ A * 2./3. * equivalentStrain(e)**(m-1.) * e[i] for i in range(len(e)) ])
    if comp == -1:
        return s
    else:
        return s[comp]

# Non-linear resolution of the stress strain relationship: find strain corresponding to the applied stress
def solveStatic(vector, *params):
    # input parameters
    p = params[0]

    # Wanted s1 s2 and e3
    s1 = p['s1']
    s2 = p['s2']
    e3 = p['e3']

    # wanted pressure
    pressure = p['pressure']

    # debug info
    iter = p['iter']
    print_debug(('Iteration: ', iter))
    p['iter']+=1

    # vector contains s1, s2, e3 of iterative resolution
    e1, e2, s3 = vector

    # stress vector
    s = [s1, s2, s3]
    #strain vector
    e = [e1, e2, e3]

    # deviatoric strain corresponding to e
    obtained_s = deviatoricStressComponent(e, p)

    # stress vector
    sigma = obtained_s - pressure*np.ones_like(obtained_s)

    # debug info
    print_debug( (sigma[0]-s1, sigma[1]-s2, sigma[2]-s3))
    print_debug(' ')

    # Objective function: stress and volume conservation(this is done by using a penalty parameter)
    if OptType == 'fsolve':
        return sigma-s+volumeConservationStress(e) * np.ones_like(s)
    elif OptType == 'leastsq' or OptType == 'nelder-mead':
        return ( (sigma[0]-s[0]+ volumeConservationStress(e))**2.+ (sigma[1]-s[1]+ volumeConservationStress(e))**2. + (sigma[2]-s[2]+ volumeConservationStress(e))**2.)**0.5


# Non-linear resolution of the boundary conditions: find the BC to satisfy Tx, mu and e3_wanted
def findSimulationBC(vector, *params):
    # input parameters
    p = params[0]

    # Goal process conditions
    tx = p['Tx']
    mu = p['mu']
    e3_wanted = p['e3']

    # debug info
    iter = p['main_iter']
    print_debug( ('Main Iteration : ', iter))
    p['main_iter']+=1

    # outer optimization: BC required
    s1, s2, pressure = vector

    ## Solve static

    #parameters for static resolution
    previous_solution = p['previous_solution']
    params_static = p.copy()
    params_static['s1'] = s1
    params_static['s2'] = s2
    params_static['pressure'] = pressure
    params_static['e3'] = e3_wanted
    params_static['iter'] = 0

    # Corresponding e1, e2, s3
    if OptType == 'fsolve':
        e1, e2, s3 =  fsolve(solveStatic, previous_solution, args=params_static, xtol=1e-06, maxfev=5000)
    elif OptType == 'leastsq':
        e1, e2, s3 =  leastsq(solveStatic, previous_solution, args=params_static)
    elif OptType == 'nelder-mead':
        res =  scipy.optimize.minimize(solveStatic, previous_solution, args=tuple([params_static]), method='nelder-mead', options={'xtol': 1e-8, 'disp': Info})
        e1, e2, s3 = res['x']
    # update of current solution to be used for the next iteration of the Static resolution
    p['previous_solution'] = (e1, e2, s3)

    # stress
    s = [s1, s2, s3]

    #strain
    e = [e1, e2, e3_wanted]

    #debug info
    print_debug( (tx-triaxiality(s) , mu-lodeAngle(s)))
    print_debug( ' ')
    print_debug( ' ')

    # convergence data: Tx, mu and pressure
    if OptType == 'fsolve':
        return (tx-triaxiality(s) , mu-lodeAngle(s), pressure-computePressure(s))
    elif OptType == 'leastsq' or OptType == 'nelder-mead':
        return ((tx-triaxiality(s))**2. + (mu-lodeAngle(s))**2.)**0.5


#Penalty on volume for Static Solution (imposing by penaty volume conservation)
def volumeConservationStress(e):
    return Penalty_stress*np.sum(e)

def computeA(K0, Temp, m1, n, eb_0, m, m4, eb):
    return K0 * exp(Temp*m1) * (eb + eb_0)**n * exp(m4/(eb + eb_0))
#    return K0 * exp(Temp*m1) * (eb + eb_0)**n

def usage():
    print "./findBC.py --Tx tx_val --mu mu_val --e3 e3_val [opts]"
    print "--K0 \t K0_val"
    print "--Temp \tTemp_val"
    print "--m1 \tm1_val"
    print "--n \tn_val"
    print "--eb_0 \teb_0_val"
    print "--m \tm_val"
    print "--m4 \tm4_val"
    print "-o, --output \tOutpu file name"
    print "-h, --help Print this message"
    print "-v, --verbose Print debug information"
    print "--ext exetension"
    print "--lib or -l Use the function as a library, returns a dictionary containing: [time, eb, epb, e[0], e[1], e[2], s[0], s[1], s[2], triaxiality, lodeAngle] "


def BCs(argv):

    # Set plotting parameters
    mpl.rc('font', family='serif', size=14, serif='Computer Modern')#'Times')
    mpl.rc('text', usetex=True)
    mpl.rc('text.latex', unicode=True)
    mpl.rc('legend', markerscale = 1., numpoints = 1)
    mpl.rc('legend', handlelength = 3, fontsize='medium')

    # Global variables
    global Info, OptType, Penalty_stress
    OptType = 'leastsq'
    OptType = 'fsolve'
    OptType = 'nelder-mead'
    Info = False
    Penalty_stress=1e8

    # Material parameters
    material = {}
    material["K0"] = 2707.0
    material["Temp"] = 1100.
    material["m1"] = -0.00325
    material["n"] = -0.135
    material["eb_0"] = 0.025
    material["m"] = 0.152
    material["m4"] =-0.055


    # Loading goals
    goals = {}
    goals["Tx_wanted"] = -1
    goals["mu_wanted"] = 1
    goals["e3_wanted"] = -0.1

    set_goals = {}
    set_goals["Tx_wanted"] = False
    set_goals["mu_wanted"] = False
    set_goals["e3_wanted"] = False


    # times
    times_data = {}
    times_data["t0"] = 0
    times_data["tf"] = 6
    times_data["dt"] = 0.5

    use_as_lib = False
    default_ouput = "SimulationBC"
    default_material = True
    default_extension = "txt"

    # Read input arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'lhvo:', ["K0=", "Temp=", "m1=", "n=", "eb_0=", "m=", "m4=", "help", "verbose", "output=", "Tx=", "mu=", "e3=", "t0=", "tf=", "dt=", "lib", "ext="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    for o, a in opts:
        if o in ("-v", "--verbose"):
            Info = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-o", "--output"):
            default_ouput = a
        elif o in ("-l", "--lib"):
            use_as_lib = True
        elif o in ("--K0", "--Temp", "--m1", "--n", "--eb_0", "--m", "--m4"):
            material[o.split("--")[1]] = float(a)
            default_material = False
        elif o in ("--Tx", "--mu", "--e3"):
            goals[o.split("--")[1]+"_wanted"] = float(a)
            set_goals[o.split("--")[1]+"_wanted"] = True
        elif o in ("--t0", "--tf", "--dt"):
            times_data[o.split("--")[1]]=float(a)
        elif o in ("--ext"):
            default_extension = a
        else:
            print "Option %s unhandled (ignored)!!" %o

    if default_material:
        print "Default material used:"
        print material

    # Check validity or input arguments
    params_ok = True
    for k in set_goals.keys():
        if not set_goals[k]:
            print "Error: option --%s not set!!" %k
            params_ok = False

    if not params_ok:
        usage()
        sys.exit(2)

    # Set parameters
    K0 = material["K0"]
    Temp = material["Temp"]
    m1 = material["m1"]
    n = material["n"]
    eb_0 = material["eb_0"]
    m = material["m"]
    m4 = material["m4"]


    # Loading goals
    Tx_wanted = goals["Tx_wanted"]
    mu_wanted = goals["mu_wanted"]
    e3_wanted = goals["e3_wanted"]

    print_debug("Material properties")
    print_debug(material)

    print_debug("Goals")
    print_debug(goals)

    # Initial guess (TODO: put something real)
    A = computeA(K0, Temp, m1, n, 0., m, m4, eb_0)
    s1, s2, pressure = (-0.5*A*e3_wanted, -0.5*A*e3_wanted, A*e3_wanted)

    # Initial guess for the corresponding static solution
    params= {'previous_solution': (-0.5*e3_wanted, -0.5*e3_wanted, A*e3_wanted)} #e1, e2, s3

    time_step = times_data["dt"]
    time_final = times_data["tf"]
    time_initial = times_data["t0"]
    dt0 = time_step/10.
    exp_m = 2.
    Constant = 1./((exp_m+1.)*(time_final-time_initial-dt0))
    s_param = np.linspace(0., 1., int((times_data["tf"] -times_data["t0"])/times_data["dt"])+1, endpoint=True)
    times = np.zeros_like(s_param)
    for i, count in zip(s_param, range(len(s_param))):
        times[count]= time_initial + dt0 * i + i ** (exp_m+1.) /((exp_m+1.)*Constant)

#    times = [0.]

    eb = 0.
    epb = 0.
    output = []
    for time, count in zip(times,range(len(times))) :

        if count:
            eb += epb * (time-times[count-1])

        A = computeA(K0, Temp, m1, n, eb_0, m, m4, eb)
        print_debug(A)


        # Set arguments for Boundary conditions optimization
        params['A'] = A
        params['m'] = m
        params['Tx'] = Tx_wanted
        params['mu'] = mu_wanted
        params['e3'] = e3_wanted
        params['main_iter'] = 0

        # Initial guess of boundary conditions
        initial_guess = (s1, s2, pressure)

        # Finding Boundary conditions
        if OptType == 'fsolve':
            s1, s2, pressure =  fsolve(findSimulationBC, initial_guess, args=params, xtol=1e-06, maxfev=5000)
        elif OptType == 'leastsq':
            s1, s2, pressure =  leastsq(findSimulationBC, initial_guess, args=params)
        elif OptType == 'nelder-mead':
            res =  scipy.optimize.minimize(findSimulationBC, initial_guess, args=tuple([params]), method='nelder-mead', options={'xtol': 1e-8, 'disp': Info})
            s1, s2, pressure = res['x']

        ## Solve static of the last resolution
        e1, e2, s3 =  params['previous_solution']

        # Strain and stress vectors
        e = [e1, e2, e3_wanted]
        s = [s1, s2, s3]
        epb = equivalentStrain(e)

        #output
        print ('the boundady conditions are: ')
        print ('stress 11: ', s1)
        print ('stress 22: ', s2)
        print ('strain rate 33: ', e3_wanted)
        print  ' *********************'
        print ('leading to: ')
        print ('strain rate 11: ', e1)
        print ('strain rate 22: ', e2)
        print ('stress 33: ', s3)
        print ('pressure: ', pressure)
        print ('epsilon bar: ', eb)
        print ('epsilon dot bat: ', epb)
        print  ' *********************'
        print  ' *********************'
        print  ' *********************'

        print ('Verification goals: ')
        print ('Wanted Tx: ', Tx_wanted, ', obtained Tx: ', triaxiality(s), ' error (%) : ', 100.*(triaxiality(s)-Tx_wanted)/Tx_wanted )
        print ('Wanted mu: ', mu_wanted, ', obtained mu: ', lodeAngle(s), ' error (%) : ', 100.*(lodeAngle(s)-mu_wanted)/(mu_wanted+(mu_wanted==0)*1.) )
        print ('Time: ', time)
#        print ('pressure obtained: ',pressure, ' computed pressure: ', computePressure(s), ' error (%) : ', 100.*(pressure-computePressure(s))/computePressure(s))
        print ('volume change: ', np.sum(e))
        print s
        if(abs((triaxiality(s)-Tx_wanted)/Tx_wanted)+abs((lodeAngle(s)-mu_wanted)/(mu_wanted+(mu_wanted==0)*1.))+abs(np.sum(e)) < 1e-4):
            print("\n**** BC FOUND!! ****\n")
            print s
            output.append([time, eb, epb, e[0], e[1], e[2], s[0], s[1], s[2], triaxiality(s), lodeAngle(s)])
        else:
            print("\n!!!! SAD DAY !!!!\n")
            break

    output = np.array(output)
    if len(default_extension.split(".")) > 1:
        default_extension = default_extension.split(".")[1]

    np.savetxt(default_ouput+"."+default_extension, output, fmt='%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f', delimiter=",")

    if use_as_lib:
        data_output = {}
        data_output['time'] = output[:,0]
        data_output['eb'] = output[:,1]
        data_output['epb'] = output[:,2]
        data_output['e0'] = output[:,3]
        data_output['e1'] = output[:,4]
        data_output['e2'] = output[:,5]
        data_output['s0'] = output[:,6]
        data_output['s1'] = output[:,7]
        data_output['s2'] = output[:,8]
        data_output['triaxiality'] = output[:,9]
        data_output['lodeAngle'] = output[:,10]
        return data_output
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(output[:,0], output[:,6],'.-', label='\sigma_1')
        ax.plot(output[:,0], output[:,7],'.-', label='\sigma_2')
        ax.legend(loc='lower right')
        ax.set_xlabel('$\mathrm{time (s)}$')
        ax.set_ylabel('$\mathrm{Stress}$')
        plt.show()
        return


if __name__ == "__main__":
    BCs(sys.argv)
