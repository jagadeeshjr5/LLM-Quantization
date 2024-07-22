import torch
import numpy as np
from sklearn.cluster import KMeans

def kmquantize(x, n_centroids=4):
    
    x_flat = torch.tensor(x.flatten().reshape(-1, 1))
    kmeans = KMeans(n_clusters=n_centroids)
    kmeans.fit(x_flat.detach().numpy())
    
    centroids = torch.tensor(kmeans.cluster_centers_)
    labels = kmeans.predict(x_flat.detach().numpy())
    labels = torch.tensor(labels.reshape(x.shape))

    quant_x = torch.tensor(centroids[labels].reshape(10, 10), requires_grad=True)

    return quant_x, centroids, labels

def finetunecentroids(x, centroids, labels, n_iters=10):

    Y = torch.tensor(centroids[labels].reshape(10, 10), requires_grad=True, dtype=x.dtype)

    for i in range(n_iters):
        loss = torch.nn.functional.mse_loss(x, Y)
        x.grad = None
        loss.backward()
    
        if x.grad is not None:
            Y = Y + x.grad

    grad_X = x.grad

    #print(grad_X)

    unique_indices = np.unique(labels)

    mean_centroids = []
    
    for index in unique_indices:
        elements = grad_X[labels == index]
        mean = torch.mean(elements)
        mean_centroids.append(mean)
    
    mean_centroids = torch.tensor(mean_centroids).reshape(-1, 1)

    mod_centroids = centroids - mean_centroids

    quant_x = torch.tensor(mod_centroids[labels].reshape(10, 10), requires_grad=True)

    return quant_x, mod_centroids 

def kmdequantize(x, centroids, labels):

    kmdequant_x = torch.tensor(centroids[labels].reshape(x.shape), dtype=x.dtype)

    return kmdequant_x