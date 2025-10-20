#%%
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import scipy

#%%
# loading the image from the disk
image = cv2.imread('../assets/image_ref.jpg')
width, height, _ = np.divide(image.shape, 1)
image = cv2.resize(image, (int(height), int(width)))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#plt.imshow(image)

# noisy image
noise = 20*np.random.normal(0, 0.6, image.shape)
N = np.clip((image + noise).astype(np.int64), 0, 255).astype(np.uint8)
plt.imshow(N)

#blurred image
cus = image.copy()
kernel = np.array([[0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1]])


kernel_base = kernel / np.sum([np.sum(i) for i in kernel])
B = np.array(cv2.filter2D(cus, -1, cv2.flip(kernel_base, -1)))
plt.imshow(B)

# denoising image
DN = cv2.fastNlMeansDenoisingColored(N, None, 10, 10, 7, 15)
plt.imshow(DN,cmap = 'gray')
#%%
def gradient(image):
    """
    Calcule la norme du gradient en chaque point d'une image numpy.
    """
    # Si l'image est en couleur, on la convertit en niveaux de gris (moyenne simple)
    if image.ndim == 3:
        image_gray = np.mean(image, axis=2)
    else:
        image_gray = image.astype(float)

    # Calcul des dérivées partielles selon x et y avec np.gradient
    gy, gx = np.gradient(image_gray)  # gy = dI/dy, gx = dI/dx

    return [gy, gx]

## Soft threshold
def soft_threshold_one_coordinate(lambd, x):
  if x > lambd :
    return x - lambd
  elif abs(x) <= lambd :
    return 0
  else:
    return x + lambd

def soft_threshold(lambd, M):
  return np.array([[soft_threshold_one_coordinate(lambd, M[i][j]) for j in range(len(M[0]))] for i in range(len(M))])

def minimizer_soft_threshold(lambd, U, rho):
  """Minimiser f(X) = lambda * || X ||_1 + rho/2 ||X - U||_2**2"""

  return soft_threshold(lambd/rho, U)

def resoud_quad_fourier(K, V):
    """
    Trouve une image im qui minimise sum_i || K_i * im - V_i ||^2
    où les K_i sont des filtres et les V_i sont des images respectivement.
    """
    n = len(K)
    assert n == len(V), "Nombre de filtres K et d'images V différent."

    sy, sx = K[0].shape

    # Initialisation propre en complexe
    numer = np.zeros((sy, sx), dtype=np.complex128)
    denom = np.zeros((sy, sx), dtype=np.float64)

    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2

    for k in range(n):
        fV = fft2(V[k])
        fK = fft2(K[k])
        numer += np.conj(fK) * fV
        denom += np.abs(fK) ** 2

    # Pour éviter une division par zéro éventuelle
    denom = np.where(denom == 0, 1e-12, denom)

    im_recon = np.real(ifft2(numer / denom))
    return im_recon

def convert_large_kernel_to_real_kernel(big_kernel, p):

    n, m = big_kernel.shape
    if p > n or p > m:
        raise ValueError(f"La taille p={p} dépasse celle du grand noyau ({n}, {m}).")

    mini = np.zeros((p, p), dtype=big_kernel.dtype)

    mid_y = (p + 1) // 2
    mid_x = (p + 1) // 2

    # 1. Petit bas-droit <- Grand haut-gauche
    mini[mid_y-1:, mid_x-1:] = big_kernel[:p - mid_y+1, :p - mid_x+1]

    # 2. Petit bas-gauche <- Grand haut-droit
    mini[mid_y-1:, :mid_x-1] = big_kernel[:p - mid_y+1, m - mid_x+1:]

    # 3. Petit haut-droit <- Grand bas-gauche
    mini[:mid_y-1, mid_x-1:] = big_kernel[n - mid_y+1:, :p - mid_x+1]

    # 4. Petit haut-gauche <- Grand bas-droit
    mini[:mid_y-1, :mid_x-1] = big_kernel[n - mid_y+1:, m - mid_x+1:]

    return mini

def convolution_circulaire(img1, img2):
    """
    Calcule la convolution circulaire 2D entre deux tableaux de même taille.
    """
    # Vérification des tailles
    if img1.shape != img2.shape:
        raise ValueError(f"Les deux images doivent avoir la même taille. "
                         f"Trouvé {img1.shape} et {img2.shape}.")

    # FFT 2D des deux images
    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)

    # Produit dans le domaine fréquentiel
    conv_hat = F1 * F2

    # Retour dans le domaine spatial
    conv_circ = np.real(np.fft.ifft2(conv_hat))

    return conv_circ

def array_to_list(array):
  liste = []
  for i in range(len(array)):
    for j in range(len(array[0])):
      liste.append(array[i][j])

  return liste

def normal_projection_kernel(kernel, taille_support):
    n, m = kernel.shape
    proj = np.zeros_like(kernel)

    N = (taille_support+1)//2
    M = taille_support - N

    elt_save_tl = array_to_list([[(i, j) for j in range(N)] for i in range(N)])
    elt_save_tr = array_to_list([[(i, j) for j in range(m-M, m)] for i in range(N)])
    elt_save_bl = array_to_list([[(i, j) for j in range(N)] for i in range(n-M, n)])
    elt_save_br = array_to_list([[(i, j) for j in range(m-M, m)] for i in range(n-M, n)])

    elt_save = elt_save_bl + elt_save_br + elt_save_tl + elt_save_tr
    for coord in elt_save:
        i,j = coord
        proj[i][j] = np.maximum(kernel[i][j], 0)

    s = proj.sum()
    if s > 0:
      proj = proj / s

    return proj

def kernel_vers_grand(k, n, m):
   p = len(k)
   N = (len(k)+1)//2
   M = len(k)-N

   out = np.zeros((n, m))
   out[0:N,0:N] =k[M:p,M:p]
   out[-M:,-M:] =k[0:M,0:M]
   out[-M:,0:N] =k[0:M,M:p]
   out[0:N,-M:] =k[M:p,0:M]
   return out / np.sum([np.sum(i) for i in out])

#%%

def admm(nabla_b, nalba_l_chapeau, lambd, taille_noyau=7, seuil=10e-8, rho=10, max_iter=100):

  n,m = nabla_b[0].shape

  u = np.zeros((n, m))
  u_new = u

  x = np.zeros((n, m))
  z = np.zeros((n, m))
  v = u

  i=0
  while  i==0 or (np.linalg.norm(u_new-u, 2) > seuil and i < max_iter):
    print(np.linalg.norm(u_new-u, 2))

    u = u_new

    x = resoud_quad_fourier([nalba_l_chapeau[0],nalba_l_chapeau[1], [[1]]], [nabla_b[0], nabla_b[1], v])
    v = z-u

    z = minimizer_soft_threshold(lambd, v, rho)
    v = x+u

    u_new = normal_projection_kernel(u + x-z, taille_noyau)

    i+= 1

  print(np.linalg.norm(u_new-u, 2))
  return u_new




def admm2(blurred, nalba_l_chapeau, kernel, lambd, beta, seuil=10e-8, rho=1, max_iter=100):

    n,m = blurred[:,:,0].shape

    latent_image = np.zeros((n, m))
    latent_image_new = latent_image
    gradient_latent_image = gradient(latent_image)

    x = np.zeros((n, m))
    z = np.zeros((n, m))
    v = latent_image

    i=0
    while  i==0 or (np.linalg.norm(latent_image_new-latent_image, 'fro') > seuil and i < max_iter):
      print(np.linalg.norm(latent_image_new-latent_image, 2))

      latent_image = latent_image_new
      gradient_latent_image = gradient(latent_image)

      kernel_derivation_x = 0.5*kernel_vers_grand(np.array([[0, 0, 0],
                                                            [1, 0, -1],
                                                            [0, 0, 0]]), n, m)

      kernel_derivation_y = 0.5*kernel_vers_grand(np.array([[0, 1, 0],
                             [0, 0, 0],
                             [0, -1, 0]]), n,m)

      x = resoud_quad_fourier([kernel, kernel, kernel, [[1]]], [blurred[:,:,0], blurred[:,:,1], blurred[:,:,1], v])
      v = z-latent_image

      z = minimizer_soft_threshold(lambd, v, rho)
      v = x+latent_image

      latent_image_new = latent_image + x-z

      i+= 1

      print(latent_image_new)

    print(np.linalg.norm(latent_image_new-latent_image, 2))
    return latent_image_new

#%%

gradient_blurred = gradient(B)
gradient_denoised = gradient(DN)
gradient_image = gradient(image)

taille_noyau = 7
kernel = admm(gradient_blurred, gradient_denoised, 0.15, seuil=1e-3 , max_iter=100, taille_noyau=taille_noyau)
kernel = kernel / np.sum([np.sum(i) for i in kernel])

noyau = convert_large_kernel_to_real_kernel(kernel, taille_noyau)
plt.imshow(noyau)
#plt.title("Noyau estimé par l'algorithme ADMM")
#plt.imsave("noyau_estimé.jpg", noyau)
plt.show()

print(noyau)