import numpy as np
import cv2

def matting(b1, b2, c1, c2):
    '''Computes the triangulation matting equation
       Param:
       b1: background image 1
       b2: background image 2
       c1: composite image 1
       c2: composite image 2
       Returns:
       fg: foreground image
       alpha: alpha image '''
    b1_r, b1_g, b1_b = b1[:, :, 0], b1[:, :, 1], b1[:, :, 2]
    b2_r, b2_g, b2_b = b2[:, :, 0], b2[:, :, 1], b2[:, :, 2]
    c1_r, c1_g, c1_b = c1[:, :, 0], c1[:, :, 1], c1[:, :, 2]
    c2_r, c2_g, c2_b = c2[:, :, 0], c2[:, :, 1], c2[:, :, 2]

    img_shape = b1.shape  # all images have same shape
    fg = np.zeros(img_shape)
    alpha = np.zeros(img_shape[:2])
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            a = np.array([[b1_r[i, j]],
                          [b1_g[i, j]],
                          [b1_b[i, j]],
                          [b2_r[i, j]],
                          [b2_g[i, j]],
                          [b2_b[i, j]]])
            b = np.array([[c1_r[i, j] - b1_r[i, j]],
                          [c1_g[i, j] - b1_g[i, j]],
                          [c1_b[i, j] - b1_b[i, j]],
                          [c2_r[i, j] - b2_r[i, j]],
                          [c2_g[i, j] - b2_g[i, j]],
                          [c2_b[i, j] - b2_b[i, j]]])
            A = np.hstack((matrix, -1 * a))
            x = np.clip(np.dot(np.linalg.pinv(A), b), 0.0, 1.0)
            fg[i, j,:] = np.array([x[0][0], x[1][0], x[2][0]])
            alpha[i, j] = x[3][0]
    return fg, alpha

def multiply_alpha(alpha, b):
    '''Multiplies (1-alpha) and the background image
       Param:
       alpha: alpha matte image
       b: new background image
       Returns:
       c: (1-alpha) * background'''
    c = np.zeros(b.shape)
    for i in range(b.shape[2]):
        c[:,:,i] = b[:,:,i] * (1-alpha)
    return c


if __name__ == '__main__':
    window = np.array(cv2.imread('window.jpg')) / 255.0
    b1 = np.array(cv2.imread('flowers-backA.jpg')) / 255.0
    b2 = np.array(cv2.imread('flowers-backB.jpg')) / 255.0
    c1 = np.array(cv2.imread('flowers-compA.jpg')) / 255.0
    c2 = np.array(cv2.imread('flowers-compB.jpg')) / 255.0
    fg, alpha = matting(b1, b2, c1, c2)
    cv2.imwrite('flowers-alpha.jpg', alpha*255.0)
    cv2.imwrite('flowers-foreground.jpg', fg*255.0)
    b = multiply_alpha(alpha, window)
    composite = fg + b
    cv2.imwrite('flowers-composite.jpg', composite * 255.0)

    b1 = np.array(cv2.imread('leaves-backA.jpg')) / 255.0
    b2 = np.array(cv2.imread('leaves-backB.jpg')) / 255.0
    c1 = np.array(cv2.imread('leaves-compA.jpg')) / 255.0
    c2 = np.array(cv2.imread('leaves-compB.jpg')) / 255.0
    fg, alpha = matting(b1, b2, c1, c2)
    cv2.imwrite('leaves-alpha.jpg', alpha*255.0)
    cv2.imwrite('leaves-foreground.jpg', fg*255.0)
    b = multiply_alpha(alpha, window)
    composite = fg + b
    cv2.imwrite('leaves-composite.jpg', composite*255.0)