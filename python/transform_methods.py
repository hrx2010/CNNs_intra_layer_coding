import numpy as np
import scipy.fftpack as fftpack

def fft2split(x):
	x = np.array(x)
	s = x.shape
	b = s[0] * s[1]
	x = np.reshape(x , (b , -1))
	x[0:int((b+1)/2-1) , :] = np.sqrt(2.0) * x[0:int((b+1)/2-1),:]
	x[int((b+1)/2):b,:] = np.sqrt(2.0)*x[int((b+1)/2):b,:]
	x[0:int((b+1)/2-1),:] = np.real(x[0:int((b+1)/2-1),:])
	x[int((b+1)/2):b,:] = np.imag(x[int((b+1)/2):b,:])
	x = np.reshape(x , s)

	return x

def ifft2split(x):
	x = np.array(x)
	s = x.shape
	b = s[0] * s[1]
	x = np.reshape(x , (b , -1))
	x[0:int((b+1)/2-1) , :] = np.sqrt(0.5) * x[0:int((b+1)/2-1),:]
	x[int((b+1)/2):b , :] = np.sqrt(0.5) * x[int((b+1)/2):b,:]
	x[0:int((b+1)/2-1) , :] = 1.0 * np.real(x[0:int((b+1)/2-1),:]) - 1j*np.real(x[b-1 : int((b+1)/2 -1) : -1,:])
	x[int((b+1)/2):b , :] = 1j*np.real(x[int((b+1)/2):b , :]) + 1.0*np.real(x[int((b+1)/2-2)::-1 , :])
	x = np.reshape(x,s);

	return x

def dft2(x):
	x = np.array(x)
	s = x.shape
	y = np.sqrt(1.0 / s[0] / s[1]) * fft2split(np.fft.fftshift(np.fft.fftshift(np.fft.fft2(x , axes=(0,1)) , 0) , 1))

	return y

def idft2(x):
	x = np.array(x)
	s = x.shape
	b = s[0] * s[1]
	y = np.fft.ifft2(np.fft.ifftshift(np.fft.ifftshift(ifft2split(np.sqrt(1.0 * b)*x),0),1) , axes=(0,1));

	t1 = ifft2split(np.sqrt(b)*x)

	return y

def dst2(x):
	x = np.array(x)
	return fftpack.dst(fftpack.dst(x,axis=0,type=1,norm='ortho'),axis=1,type=1,norm='ortho')

def idst2(x):
	x = np.array(x)
	return fftpack.idst(fftpack.idst(x,axis=1,type=1,norm='ortho'),axis=0,type=1,norm='ortho')

def dct2(x):
	x = np.array(x)
	return fftpack.dct(fftpack.dct(x,axis=0,type=1,norm='ortho'),axis=1,type=1,norm='ortho')

def idct2(x):
	x = np.array(x)
	return fftpack.idct(fftpack.idct(x,axis=1,type=1,norm='ortho'),axis=0,type=1,norm='ortho')

transforms = [[dct2,idct2,dft2],[dst2,idst2,idft2]]
