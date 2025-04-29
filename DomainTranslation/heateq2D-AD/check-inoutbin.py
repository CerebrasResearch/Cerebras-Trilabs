import numpy as np
import sys

#python ../make-inputbin.py $(NX) nlocal.bin data.bin

nx = int(sys.argv[1])
ny = int(sys.argv[2])
n = int(sys.argv[3])
p = int(sys.argv[4])

nx2 = nx//2
ny2 = ny//2
w = 2*p

nxtot = nx*n
nytot = ny*n


paramsfile = sys.argv[5]
x0file = sys.argv[6]
xfile = sys.argv[7]
yfile = sys.argv[8]

print("check-inoutbin: nx=%d, ny=%d, n=%d, p=%d" % (nx,ny,n,p))

param = np.fromfile(paramsfile,np.float32).reshape([nx2,ny2,7])

#struct paramstruct {
#  sp h;  // grid spacing
#  sp dt; // time step
#  sp D;  // diffusion coefficient
#  sp niter;
#};


# Set up internal parameters
(origin,h,dt,D,niter,ninner,nsample) = param[0,0,:]
print("Parameters h=%.3f dt=%.3f D=%.3f niter=%.3f ninner=%.3f nsample=%.3f" % \
      (h,dt,D,niter,ninner,nsample))
if int(niter) != niter:
    raise Exception("niter=%.16e not an integer!" % niter)
if int(ninner) != ninner:
    raise Exception("ninner=%.16e not an integer!" % ninner)
if int(nsample) != nsample:
    raise Exception("nsample=%.16e not an integer!" % nsample)
niter = int(niter*ninner)

def nstr(f):
    if f == 0:
        return "     ";
    else:
        return "%5.1f" % np.log10(f)

    
def mprint(x):
    for row in x:
        s = " ".join(["%8.4f" % t for t in row])
        print(s)

def globfromfile(fname):
    pos00 = np.fromfile(fname + ".00",np.float32).reshape([ny2,nx2,n+w,n+w])
    pos01 = np.fromfile(fname + ".01",np.float32).reshape([ny2,nx2,n+w,n+w])
    pos10 = np.fromfile(fname + ".10",np.float32).reshape([ny2,nx2,n+w,n+w])
    pos11 = np.fromfile(fname + ".11",np.float32).reshape([ny2,nx2,n+w,n+w])

    upad = np.zeros([ny,(n+w),nx,(n+w)])
    upad[0::2,:,0::2,:] = pos00.transpose([0,2,1,3])
    upad[1::2,:,0::2,:] = pos01.transpose([0,2,1,3])
    upad[0::2,:,1::2,:] = pos10.transpose([0,2,1,3])
    upad[1::2,:,1::2,:] = pos11.transpose([0,2,1,3])
    upadglob = upad.reshape([ny*(n+w),nx*(n+w)])
    uglob = upad[:,w:(n+w),:,w:(n+w)].reshape([nytot,nxtot])

    return (upadglob,uglob)

(pospad,inx0) = globfromfile(x0file)
(outpad,outx) = globfromfile(xfile)

print("inx0")
mprint(inx0)
print("outx")
mprint(outx)


x = np.zeros([ny*n+w,nx*n+w])
x[w:(nytot+w),w:(nxtot+w)] = inx0
y = np.zeros(x.shape)

a = D*dt/h**2

dx = [0,-1,1, 0,0]
dy = [0, 0,0,-1,1]
wgt  = [a*t for t in [-4,1,1,1,1]]
wgt[0] += 1
nwgt = len(wgt)

for iter in range(niter):
    (xptr,yptr) = (x,y)
    
    xptr[:,0:w] = xptr[:,nxtot:(nxtot+w)]
    xptr[0:w,:] = xptr[nytot:(nytot+w),:]
    yptr[w:(nytot+w),w:(nxtot+w)] = 0
    for i in range(nwgt):
        yptr[w:(nytot+w),w:(nxtot+w)] += \
            wgt[i]*xptr[(p+dy[i]):(nytot+p+dy[i]),(p+dx[i]):(nxtot+p+dx[i])]

    xptr[w:(nytot+w),w:(nxtot+w)] = yptr[w:(nytot+w),w:(nxtot+w)]


nrmx = np.linalg.norm(outx - x[w:(nytot+w),w:(nxtot+w)])

rex = nrmx / np.linalg.norm(x[w:(nytot+w),w:(nxtot+w)])
print("norm outx - x = %.3e" % nrmx)
print("rel err x = %.3e" % rex)

tol = 1e-6
if rex > tol:
    raise Exception("In accurate solution")
