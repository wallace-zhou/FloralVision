import scipy.io as sio

mat_file = sio.loadmat('imagelabels.mat')
label = mat_file['labels']
print(label)