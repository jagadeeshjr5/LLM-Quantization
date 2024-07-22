import numpy as np

def scale(rmax, rmin, bits=8, mode='affine'):
    if mode == 'affine':
        qmax = 2**(bits - 1) - 1
        qmin = -2**(bits - 1)
        s = (rmax - rmin) / (qmax - qmin)
    else:
        s = rmax / 2**(bits - 1)
    return s

def zero_point(s, rmin, bits=8, mode='affine'):
    if mode == 'affine':
        qmax = 2**(bits - 1) - 1
        qmin = -2**(bits - 1)
        z = np.round(qmin - (rmin / s))
    else:
        z = 0
    return z

def quantize(x, bits=8, mode='affine', dtype=np.int8):
    s = scale(x.max(), x.min(), bits)
    z = zero_point(s, x.min(), bits)
    qmax = 2**(bits - 1) - 1
    qmin = -2**(bits - 1)
    quantized_x = np.round(1 / s * x + z, decimals=0)
    quantized_x = np.clip(quantized_x, a_min=qmin, a_max=qmax)
    quantized_x = quantized_x.astype(dtype)
    return quantized_x

def dequantize(qx, x, bits=8, mode='affine', dtype=np.float32):
    s = scale(x.max(), x.min(), bits)
    z = zero_point(s, x.min(), bits)
    qx = qx.astype(np.int32)
    dequantized_x = s * (qx - z)
    dequantized_x = dequantized_x.astype(dtype)
    return dequantized_x

def linear_quantized_matrix_multiplication(X, W, b, bits=8):

    p = W.shape[0]

    qmax = 2**(bits - 1) - 1
    qmin = -2**(bits - 1)

    Y = X@W

    sW = scale(W.max(), W.min())
    sX = scale(X.max(), X.min())
    sb = scale(b.max(), b.min())
    sY = scale(Y.max(), Y.min())

    zW = zero_point(sW, W.min())
    zX = zero_point(sX, X.min())
    zb = zero_point(sb, b.min())
    zY = zero_point(sY, Y.min())

    qX = quantize(X)
    qW = quantize(W)
    qb = quantize(b)
    
    quantized_Y = ((zY + (sb / sY * (qb.astype(np.int32) - zb)) + (sW * sX / sY) * (((qW.astype(np.int32) @ qX.astype(np.int32)) - (zW*qX.astype(np.int32)) - (zX * qW.astype(np.int32)) + (zW * zX)).astype(np.int32) + zY).astype(np.int32)))
        
    quantized_Y = np.round(quantized_Y, decimals=0)
    quantized_Y = np.clip(quantized_Y, a_min=qmin, a_max=qmax)
    quantized_Y = quantized_Y.astype(np.int8)
    
    return quantized_Y