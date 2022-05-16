import msprime
import matplotlib.pyplot as plt
from IPython.display import display, SVG
import numpy as np
from PIL import Image
from scipy import stats


#For plotting purposes
plt.rcParams.update({'font.size': 30})

#General Variables
mut_rate = 0.001
recom_rate = 0.001
genome_length = 100000

#Create and return tree sequence
def generate_genotype(N, mut_rate, recom_rate, length, Ne, seed):  ##N = sample size, mut_rate = Mutation rate per generation
    ts = msprime.simulate(N, Ne = Ne, mutation_rate = mut_rate, recombination_rate = recom_rate,
                          length = length, random_seed = seed)
    return ts

#Show trees (will only work for small trees)
def generate_tree_svg(ts): #creates svg plot of tree sequence
    display(SVG(ts.draw_svg()))
    
#Find recombination spots
def find_breaks(ts):
    for tree in ts.trees():
        print(f"Tree {tree.index} covers {tree.interval}")
        if tree.index >= 4:
            print("...")
            break
    print(f"Tree {ts.last().index} covers {ts.last().interval}")

#Find mutations
def find_mutations(ts, list_mut = False):
    print(f"{ts.num_mutations} mutation(s) in the tree sequence:")
    if list_mut:
        for mutation in ts.mutations():
            print(ts.tables.sites)
        
#Visualise SNP data matrix X
def visualise_mat(X, scale): #scale = pixels per matrix entry
    img = Image.new('1', tuple(np.multiply(X.shape, scale)))
    pixels = img.load()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for x in range(scale):
                for y in range(scale):
                    pixels[scale*i+x,scale*j+y] = int(X[i][j])
    img.show()

#Show "LD" matrix
def show_corr_mat(X):
    plt.matshow(abs(np.corrcoef(X)), cmap = 'Reds')
    plt.show()

#Find SNP frequency for eac SNP
def find_freq(X): #find SNP frequency
    N = len(X[0])
    f = [0 for i in range(len(X))]
    for i in range(len(X)):
        f[i] = np.sum(X[i])/N
    return f

#Remove SNPs with freqeunce below `threshold`
def remove_low_f(X, threshold, mut_loc):
    f = find_freq(X)
    if threshold == 0:
        return X, f, mut_loc
    else:
        vals = []
        for i in range(len(f)):
            if f[i] < threshold:
                vals = np.append(vals, int(i))
        X = np.delete(X, list(map(int, vals)), 0)
        mut_loc = np.delete(mut_loc, list(map(int, vals)), 0)
        f_new = find_freq(X)
        return X, f_new, mut_loc

#Randomly chose causal set, and randomly assign effect sizes
def generate_func_B(X, mut_loc, func_number, freq_threshold = 0): #returns B and list of functional SNPs
    X, f = remove_low_f(X, freq_threshold, mut_loc)[0:2]
    B = np.zeros(len(f))
    func_B = np.random.choice([i for i in range(len(f))],
                              size = func_number, replace = False)
    for i in func_B:
        sd = np.power((f[i]*(1-f[i])),(1/2))
        B[i] = np.random.normal(loc = 0.0, scale = sd)
    return B, func_B
    
#Generate "traits" given X, B and sd (where sd may be scaled in scaled setting)
def generate_y(X, B, sd): #generates a "sample Y from X and B"
    N = len(X[0])
    Y = np.zeros(N)
    for i in range(N):
        Y[i] = np.dot(X[:,i],B.transpose())  + np.random.normal(loc = 0, scale = sd)
    return Y

#Perform GWAS, returns Bhat, Std. Err., and p values
def lin_regress(X,Y):
    L = len(X)
    B_hat = np.zeros(L)
    B_hat_se = np.zeros(L)
    p_vals = np.zeros(L)
    for i in range(L):
        B_hat[i], intercept, r_value, p_vals[i], B_hat_se[i] = stats.linregress(X[i,:],Y)
    return B_hat, B_hat_se, p_vals

#Visualise p value graph
def show_p_val_graph(p_vals, B, p, N):
    plt.figure(figsize = (3*6.4, 3*4.8))
    plt.scatter(np.arange(len(p_vals)), -np.log10(p_vals), marker = "o", s=16)
    for i in range(len(p_vals)):
        if B[i] != 0:  ##could be more efficient
            plt.axvline(x=i, color = 'red', lw = 2)
    plt.hlines(y = -np.log10(0.05/N),xmin = 0, xmax = len(B),
               linestyles='--', color = 'black', lw = 2)
    plt.xlabel("SNP number")
    plt.ylabel("-log10(p-value)")
    plt.title("p-value plot")
    plt.show()

#Find average LD score at a given distance `dist` 
def LDscore(X, mut_loc, dist):
    array = mut_loc
    l = len(array)
    idxL = np.zeros(l-1)
    idxR = np.zeros(l-1)
    for i in range(l-1):
        idxL[i] = (np.abs(array - array[i+1] + dist)).argmin()
        idxR[i] = (np.abs(array - array[i] - dist)).argmin()
        
    LDcoef = np.zeros(l-2)
    for i in range(l-2):
        LDcoefL = (np.corrcoef(X[i+1], X[int(idxL[i])])[0,1])**2
        LDcoefR = (np.corrcoef(X[i+1], X[int(idxR[i])])[0,1])**2
        distL = np.abs(array[i+1] - array[int(idxL[i])] - dist)
        distR = np.abs(array[i+1] - array[int(idxR[i])] - dist)
        LDcoef[i] = (distR/(distL+distR))*LDcoefL + (distL/(distL+distR))*LDcoefR
    return np.mean(LDcoef)

#Empirical estimate of MSE of B against B_hat (or B_tilde). Returns list of MSE contributions from each SNP
def find_MSE(B, B_hat):
    MSE = np.square(np.subtract(B,B_hat))
    return MSE

#Find B tilde, also returns R, the transformation matrix
def find_B_tilde(X, B):
    Rmat = np.cov(X)
    for i in range(len(Rmat)):
        Rmat[i,:] = Rmat[i,:]/Rmat[i,i]
    B_tilde = np.matmul(Rmat, B)
    return B_tilde, Rmat
            
#Generate everythin we need, in the scaled setting
def generate_scaled_sample(N, n, mut_rate, recom_rate, genome_length, freq_threshold, func_number, seed):
    scale = N/n
    #print("*")
    ts = generate_genotype(n, scale*mut_rate, scale*recom_rate, genome_length, 1/scale, seed)
    X = ts.genotype_matrix()
    mut_loc = ts.tables.sites.position
    X, f, mut_loc = remove_low_f(X, freq_threshold, mut_loc)
    B, B_func = generate_func_B(X, mut_loc, func_number)
    Y = generate_y(X, B, 1/np.sqrt(scale))
    B_hat, B_hat_se, pvals = lin_regress(X, Y)
    B_tilde, Rmat = find_B_tilde(X, B)
    MSE_B_tilde = find_MSE(B_tilde, B_hat)
    MSE_B = find_MSE(B, B_hat)
    return X, Y, f, mut_loc, B, B_func, B_hat, B_hat_se, B_tilde, Rmat, pvals, MSE_B, MSE_B_tilde, ts

#Generate everything we need in the unscaled setting
def generate_sample(N, mut_rate, recom_rate, genome_length, freq_threshold, func_number, seed):
    #print("*")
    ts = generate_genotype(N, mut_rate, recom_rate, genome_length, 1, seed)
    X = ts.genotype_matrix()
    mut_loc = ts.tables.sites.position
    X, f, mut_loc = remove_low_f(X, freq_threshold, mut_loc)
    B, B_func = generate_func_B(X, mut_loc, func_number)
    Y = generate_y(X, B, 1)
    B_hat, B_hat_se, pvals = lin_regress(X, Y)
    B_tilde, Rmat = find_B_tilde(X, B)
    MSE_B_tilde = find_MSE(B_tilde, B_hat)
    MSE_B = find_MSE(B, B_hat)
    return X, Y, f, mut_loc, B, B_func, B_hat, B_hat_se, B_tilde, Rmat, pvals, MSE_B, MSE_B_tilde, ts

#Visualise MSE B vs B tilde, returns arrays containing avg MSE and Ns
def MSEB_vs_MSEBtilde(Nstart, Nmax, Nstep, n, n_scaled, func_number):
    Ns = np.arange(Nstart, Nmax + Nstep, Nstep)
    MSElist1 = np.zeros(len(Ns))
    MSElist2 = np.zeros(len(Ns))

    for i in range(len(Ns)):
        MSEmean1 = 0
        MSEmean2 = 0
        for j in range(n):
            print(i,j)
            sample = generate_scaled_sample(Ns[i], n_scaled, mut_rate, recom_rate,
                                            genome_length, 0.02, func_number, seed = np.random.randint(1,100000))
            MSEmean1 += np.mean(sample[-3])
            MSEmean2 += np.mean(sample[-2])
        MSElist1[i] = MSEmean1/n
        MSElist2[i] = MSEmean2/n
            
    plt.figure(figsize = (3*6.4,3*4.8))
    plt.plot(Ns, MSElist1, label = "MSE against B", lw = 4)
    plt.plot(Ns, MSElist2, label = "MSE against B tilde", lw = 4)
    plt.ylim(bottom = 0)
    plt.xlabel("Effective sample size N")
    plt.ylabel("MSE of B hat against B and B tilde")
    plt.title("Convergence to B vs B tilde")
    plt.legend()
    plt.show()
    return MSElist1, MSElist2, Ns
  
#Visualise MSE B tilde vs N in scaled setting, Also returns arrays of MSE and corresponding N    
def MSEBtilde_vs_N(Nstart, Nmax, Nstep, n, n_scaled, func_number, log = True):
    Ns = np.arange(Nstart, Nmax + Nstep, Nstep)
    Ns = np.array([10**(i/2) for i in range(21)])
    MSElist = np.zeros(len(Ns))
    print(len(Ns))

    for i in range(len(Ns)):
        MSEmean = 0

        for j in range(n):
            print(i,j)
            sample = generate_scaled_sample(Ns[i], n_scaled, mut_rate, recom_rate,
                                            genome_length, 0.02, func_number, seed = np.random.randint(1,100000))
            MSEmean += np.mean(sample[-2])
        MSElist[i] = MSEmean/n

            
    plt.figure(figsize = (3*6.4,3*4.8))
    plt.plot(Ns, MSElist, label = "Convergence to B", lw = 4)
    if log:
        plt.xscale('log')
        plt.yscale('log')
        plt.title("log log plot of N vs MSE")
    else:
        plt.title("Plot of N vs MSE")
        plt.ylim(bottom = 0)
    plt.xlabel("Effective sample size N")
    plt.ylabel("MSE")
    plt.show()
    
    return Ns, MSElist

#Visualise MSE B vs N in scaled setting, Also returns arrays of MSE and corresponding N    
def MSEB_vs_N(Nstart, Nmax, Nstep, n, n_scaled, func_number, log = True):
    Ns = np.arange(Nstart, Nmax + Nstep, Nstep)
    MSElist = np.zeros(len(Ns))
    print(len(Ns))

    for i in range(len(Ns)):
        MSEmean = 0

        for j in range(n):
            print(i,j)
            sample = generate_scaled_sample(Ns[i], n_scaled, mut_rate, recom_rate,
                                            genome_length, 0.02, func_number, seed = np.random.randint(1,100000))
            MSEmean += np.mean(sample[-3])
        MSElist[i] = MSEmean/n

            
    plt.figure(figsize = (3*6.4,3*4.8))
    plt.plot(Ns, MSElist, label = "Convergence to B", lw = 4)
    if log:
        plt.xscale('log')
        plt.yscale('log')
        plt.title("log log plot of N vs MSE")
    else:
        plt.title("Plot of N vs MSE")
        plt.ylim(bottom = 0)
    plt.xlabel("Effective sample size N")
    plt.ylabel("MSE")
    plt.show()
    
    return Ns, MSElist











