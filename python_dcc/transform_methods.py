import numpy as np
import scipy.fftpack as fftpack
import scipy.io as sio
import tensorflow as tf

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
	#x = np.array(x)
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

#transforms = {'dct2' : [dct2,idct2], 'dst2' : [dst2,idst2], 'dft2': [dft2,idft2]}

def load_transform_data_from_file(file_path):
	transform_data = sio.loadmat(file_path)
	for key in transform_data:
		transform_matrices = transform_data[key]
	return transform_matrices


def transform_given_matrices(transform_matrices , x):
	
	[dim1 , dim2] = transform_matrices.shape
	[fh , fw , n_input , n_output] = x.shape
	y = np.zeros(x.shape)
	
	if dim2 == 1:
		for i in range(n_output):
			for j in range(n_input):
				H = x[: , : , j , i]
				H_vectorized = H.T.flatten()
				W = np.dot(transform_matrices[j , 0] , H_vectorized)
				W_reshaped = np.reshape(W , H.shape)
				W_reshaped = W_reshaped.T
				y[: , : , j , i] = W_reshaped

		return y

	elif dim2 == 2:
		half_n_output = int(n_output / 2)
		
		for i in range(half_n_output):
			for j in range(n_input):
				H = x[: , : , j , i]
				H_vectorized = H.T.flatten()
				W = np.dot(transform_matrices[j , 0] , H_vectorized)
				W_reshaped = np.reshape(W , H.shape)
				W_reshaped = W_reshaped.T
				y[: , : , j , i] = W_reshaped

		for i in range(half_n_output):
			for j in range(n_input):
				H = x[: , : , j , i + half_n_output]
				H_vectorized = H.T.flatten()
				W = np.dot(transform_matrices[j , 1] , H_vectorized)
				W_reshaped = np.reshape(W , H.shape)
				W_reshaped = W_reshaped.T
				y[: , : , j , i + half_n_output] = W_reshaped

		return y


def i_transform_given_matrices_fast(inv_transform_matrices_reshaped , x , sess):
	[fh , fw , n_input , n_output] = x.shape
	x_reshaped = np.zeros((n_input * n_output , fh * fw , 1))

	cnt = 0

	for j in range(n_output):
		for i in range(n_input):
			x_reshaped[cnt , : , 0] = x[: , : , i , j].T.flatten()
			cnt = cnt + 1

	A = tf.constant(inv_transform_matrices_reshaped)
	B = tf.constant(x_reshaped)
	C = tf.matmul(A, B)

	y = sess.run(C)

	rst = np.zeros([fh , fw , n_input , n_output])

	cnt = 0

	for j in range(n_output):
		for i in range(n_input):
			rst[: , : , i , j] = np.reshape( y[cnt , : , :], (fh , fw) ).T
			cnt = cnt + 1

	return rst
