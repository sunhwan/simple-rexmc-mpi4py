from StringIO import StringIO
import sys, os
import numpy as np

os.environ["CC"] = "gcc-5"
os.environ["CXX"] = "g++-5"

debug = False
n_max = False
if len(sys.argv) > 1: n_max = int(sys.argv[1])

#input = sys.stdin
#pmf_filename = input.readline().strip() # stores pmf
#rho_filename = input.readline().strip() # stores average density
#bia_filename = input.readline().strip() # stores biased distribution
#fff_filename = input.readline().strip() # stores F(i)
#temperature = float(input.readline().strip())
pmf_filename = 'run.pmf'
rho_filename = 'run.rho'
bia_filaname = 'run.bia'
fff_filename = 'run.fff'
temperature = 300.

#xmin, xmax, deltax, is_x_periodic = map(float, input.readline().strip().split())
#ymin, ymax, deltay, is_y_periodic = map(float, input.readline().strip().split())
#nwin, niter, fifreq  = map(int, input.readline().strip().split())
#tol = map(float, input.readline().strip().split())
xmin, xmax, deltax, is_x_periodic = -6, 7, 0.2, 0
ymin, ymax, deltay, is_y_periodic = -180.0, 60, 1., 1
nwin, niter, fifreq = 294, 10000, 10
tol = 1e-3
is_x_periodic = bool(is_x_periodic)
is_y_periodic = bool(is_y_periodic)
nbinx = int((xmax - xmin) / deltax + 0.5)
nbiny = int((ymax - ymin) / deltay + 0.5)
kb = 0.0019872
kbt = kb * temperature
beta0 = 1.0/kbt

if debug:
    temperature = 283.15
    kbt = kb * temperature
    beta0 = 1.0/kbt

k1 = np.zeros(nwin)
cx1 = np.zeros(nwin)
k2 = np.zeros(nwin)
cx2 = np.zeros(nwin)
hist = np.zeros((nwin, nbinx, nbiny), dtype=np.int)
nb_data = np.zeros(nwin, dtype=np.int)
x1 = lambda j: xmin + (j+1)*deltax - 0.5*deltax
y1 = lambda j: ymin + (j+1)*deltay - 0.5*deltay
press = 1.01325 * 1.4383 * 10**-5


data_range = [[None, None], [None, None], [None, None], [None, None]]

fp = open('metafile6')
for i in range(nwin):
    #line = input.readline().strip()
    line = fp.readline().strip()
    fname = line.split()[0]
    cx1[i], cx2[i], k1[i], k2[i] = map(float, line.split()[1:5])

    def mkhist(fname, ihist, k1, cx1, k2, cx2):
        xdata = []
        ydata = []
        count = 0
        for l,line in enumerate(open(fname)):
            if line.startswith("#"): continue
            #if l < 5000: continue
            time, x, y = map(float, line.strip().split()[:3])
            xdata.append(x)
            ydata.append(y)
            if debug and len(xdata) > 10000: break
            if n_max and len(xdata) > n_max: break
        x = np.array(xdata)
        y = np.array(ydata)
        xbins = [xmin+i*deltax for i in range(nbinx+1)]
        ybins = [ymin+i*deltay for i in range(nbiny+1)]
        data = np.array((x,y)).transpose()

        hist[ihist], edges = np.histogramdd(data, bins=(xbins, ybins), range=((xmin, xmax), (ymin, ymax)))
        nb_data[ihist] = np.sum(hist[ihist])

        for k in range(2):
            t = (x,y)[k]
            if data_range[k][0] is None or np.min(t) < data_range[k][0]: data_range[k][0] = np.min(t)
            if data_range[k][1] is None or np.max(t) > data_range[k][1]: data_range[k][1] = np.max(t)

        print 'statistics for timeseries # ', ihist, fname
        print 'minx:', '%8.3f' % np.min(x), 'maxx:', '%8.3f' % np.max(x)
        print 'average x', '%8.3f' % np.average(x), 'rms x', '%8.3f' % np.std(x)
        print 'miny:', '%8.3f' % np.min(y), 'maxy:', '%8.3f' % np.max(y)
        print 'average y', '%8.3f' % np.average(y), 'rms x', '%8.3f' % np.std(y)
        print

    mkhist(fname, i, k1[i], cx1[i], k2[i], cx2[i])

print 'minx:', '%8.3f' % data_range[0][0], 'maxx:', '%8.3f' % data_range[0][1]
print 'miny:', '%8.3f' % data_range[1][0], 'maxy:', '%8.3f' % data_range[1][1]

print hist.shape
print nb_data

## write biased distribution
#f = open(bia_filename, 'w')
#for j in range(nbinx):
#    for k in range(nbiny):
#        for k in range(nbinu):
#            f.write("%8d\n" % np.sum(hist[:,:,j,k]))

# iterate wham equation to unbias and recombine the histogram
TOP = np.zeros((nbinx, nbiny), dtype=np.int32)
BOT = np.zeros((nbinx, nbiny))
W1 = np.zeros((nwin, nbinx, nbiny))

for i in range(nwin):
    for k in range(nbinx):
        for l in range(nbiny):
            W1[i,k,l] = k1[i]*(x1(k) - cx1[i])**2 + k2[i]*(y1(l) - cx2[i])**2

TOP = np.sum(hist, axis=0)
np.set_printoptions(linewidth=200)

from scipy import weave
from scipy.weave import converters

def wham2d(nb_data, TOP, nbinx, nbiny, W1, beta0, F=None):
    icycle = 1
    rho = np.zeros((nbinx, nbiny), np.double)
    if F is None: F = np.zeros(nwin)
    F2 = np.zeros(nwin, np.double)

    while icycle < niter:
        code_pragma = """
            #pragma omp parallel num_threads(nthreads)
            {
                #pragma omp for collapse(2) 
                for (int k=0; k<nbinx; k++) {
                    for (int l=0; l<nbiny; l++) {
                        double BOT = 0.0;
                        for (int i=0; i<nwin; i++) {
                            BOT += nb_data(i)*exp(F(i)-beta0*W1(i,k,l));
                        }

                        if (BOT < 1e-100 || TOP(k,l) == 0) continue;
                        rho(k,l) = TOP(k,l) / BOT;
                    }
                }

                #pragma omp for collapse(1)
                for (int i=0; i<nwin; i++) {
                    for (int k=0; k<nbinx; k++) {
                        for (int l=0; l<nbiny; l++) {
                            F2(i) += rho(k,l)*exp(-beta0*W1(i,k,l)); 
                        }
                    }
                }
            }
        """

        nthreads = 4
        weave.inline(code_pragma, ['F', 'F2', 'rho', 'nb_data', 'beta0', 'W1', 'TOP', 'nbinx', 'nbiny','nwin', 'nthreads'], type_converters=converters.blitz, extra_compile_args=['-O3 -fopenmp'], extra_link_args=['-O3 -fopenmp'], headers=['<omp.h>'])#, library_dirs=['/Users/sunhwan/local/python/lib'])

        converged = True
        F2 = -np.log(F2)
        F2 = F2 -np.min(F2)

        diff = np.max(np.abs(F2 - F))

        if diff > tol: converged = False
        print 'round = ', icycle, 'diff = ', diff
        icycle += 1

        if ( fifreq != 0 and icycle % fifreq == 0 ) or ( icycle == niter or converged ):
           #open(fff_filename, 'w').write("%8i %s\n" % (icycle, " ".join(["%8.3f" % f for f in F2]))) 
           if icycle == niter or converged: break

        F = F2
        F2 = np.zeros(nwin)

    return F2, rho

F = np.zeros(nwin)
if i == 0 and os.path.exists(fff):
    F = np.loadtxt(fff)
F, rho = wham2d(nb_data, TOP, nbinx, nbiny, W1, beta0, F)
np.savetxt(fff_filename, F)

# jacobian (distance)
#for j in range(nbinx):
#    rho[j] = rho[j] / x1(j)**2

# jacobian (plane)
#for j in range(nbiny):
#    rho[:,j] = rho[:,j] / y1(j)

fp = open(rho_filename, 'w')
for j in range(rho.shape[0]):
    for k in range(rho.shape[1]):
        fp.write("%8f %8f %8f\n" % (x1(j), y1(k), rho[j,k]))
    fp.write("\n")

jmin = np.argmax(rho)
rhomax = rho[np.unravel_index(jmin, rho.shape)]

np.seterr(divide='ignore')
pmf = -kbt * np.log(rho/rhomax)

fp = open(pmf_filename, 'w')
for j in range(rho.shape[0]):
    for k in range(rho.shape[1]):
        fp.write("%8f %8f %8f\n" % (x1(j), y1(k), pmf[j,k]))
    fp.write("\n")
