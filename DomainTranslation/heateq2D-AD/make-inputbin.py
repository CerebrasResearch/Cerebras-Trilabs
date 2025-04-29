import numpy as np
import sys

#python ../make-inputbin.py $(NX) nlocal.bin data.bin

skip_x0 = True

nx = int(sys.argv[1])
ny = int(sys.argv[2])
n = int(sys.argv[3])
p = int(sys.argv[4])

paramsfile = sys.argv[5]
posfile = sys.argv[6]

origin = float(sys.argv[7])

def mprint(x):
    for row in x:
        s = " ".join(["%8.5f" % t for t in row])
        print(s)


print("make-inputbin: nx=%d, ny=%d, n=%d, p=%d, origin=%.3e" % \
      (nx,ny,n,p,origin))


if nx%2 != 0 or ny%2 != 0:
    raise Exception("nx=%d and ny=%d must be even!" % (nx,ny))


#struct paramstruct {
#  sp h;  // grid spacing
#  sp dt; // time step
#  sp D;  // diffusion coefficient
#  sp niter;
#};

h = 1.0
D = 1.0
dt = 1.0/64.0
niter = 256
ninner = 1000
nsample = 1

e1 = np.zeros((nx+ny)//2)
e1[0] = 1;
params = np.array([ [ [origin*e1[ix]*e1[iy],h,dt,D,niter,ninner,nsample] \
                      for iy in range(ny//2) ] \
                    for ix in range(nx//2) ])
#print(params)

print("params shape")
print(params.shape)

nxtot = n*nx
nytot = n*ny
w = 2*p

params.astype('float32').tofile(paramsfile)

if skip_x0:
    print("Not writing x0.bin files.")
    sys.exit(0)
    
case = 3
if case == 1:
    # Each element value is its index.
    uglob = np.array([ [ i*nxtot+j+1 for j in range(nxtot) ] \
                       for i in range(nytot) ])
elif case == 2:
    # Random input array
    uglob = np.random.rand(nytot,nxtot)
elif case == 3:
    # delta function at (0,0)
    uglob = np.zeros([nytot,nxtot])
    uglob[0,0] = origin
    uglob[n,0] = origin
    uglob[0,n] = origin
    uglob[n,n] = origin
else:
    raise Exception("Unknown initial value case %d" % case)
    
#print("uglob")
#mprint(uglob)

uloc = np.zeros([ny,nx,n+w,n+w])
uloc[:,:,w:(n+w),w:(n+w)] = uglob.reshape([ny,n,nx,n]).transpose([0,2,1,3])
uloc[:,:,0:w,:] = -3
uloc[:,:,:,0:w] = -5
uloc[:,:,0:w,0:w] = -7

print("uloc shape")
print(uloc.shape)

pos00 = uloc[0::2,0::2,:,:]
pos01 = uloc[1::2,0::2,:,:]
pos10 = uloc[0::2,1::2,:,:]
pos11 = uloc[1::2,1::2,:,:]

print("pos00 shape")
print(pos00.shape)

pos00.astype('float32').tofile(posfile + ".00")
pos01.astype('float32').tofile(posfile + ".01")
pos10.astype('float32').tofile(posfile + ".10")
pos11.astype('float32').tofile(posfile + ".11")
