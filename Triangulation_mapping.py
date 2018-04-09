import numpy as np
import cv2

def matting(back1, back2, comp1, comp2):
    '''Computes the triangulation matting equation
       Param:
       back1: background image 1
       back2: background image 2
       comp1: composite image 1
       comp2: composite image 2
       Returns:
       fg: foreground image
       alpha: alpha image '''
    back1_b, back1_g, back1_r = back1[:, :, 0], back1[:, :, 1], back1[:, :, 2]
    back2_b, back2_g, back2_r = back2[:, :, 0], back2[:, :, 1], back2[:, :, 2]
    comp1_b, comp1_g, comp1_r = comp1[:, :, 0], comp1[:, :, 1], comp1[:, :, 2]
    comp2_b, comp2_g, comp2_r = comp2[:, :, 0], comp2[:, :, 1], comp2[:, :, 2]

    img_shape = back1.shape  # all images have same shape
    fg = np.zeros(img_shape)
    alpha = np.zeros(img_shape[:2])
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            a = np.array([[back1_b[i, j]],
                          [back1_g[i, j]],
                          [back1_r[i, j]],
                          [back2_b[i, j]],
                          [back2_g[i, j]],
                          [back2_r[i, j]]])
            b = np.array([[comp1_b[i, j] - back1_b[i, j]],
                          [comp1_g[i, j] - back1_g[i, j]],
                          [comp1_r[i, j] - back1_r[i, j]],
                          [comp2_b[i, j] - back2_b[i, j]],
                          [comp2_g[i, j] - back2_g[i, j]],
                          [comp2_r[i, j] - back2_r[i, j]]])
            A = np.hstack((matrix, -1 * a))
            x = np.clip(np.dot(np.linalg.pinv(A), b), 0.0, 1.0)
            fg[i, j,:] = np.array([x[0][0], x[1][0], x[2][0]])
            alpha[i, j] = x[3][0]
    return fg, alpha

def multiply_alpha(alpha, back):
    '''Multiplies (1-alpha) and the background image
       Param:
       alpha: alpha matte image
       back: new background image
       Returns:
       c: (1-alpha) * background'''
    c = np.zeros(back.shape)
    for i in range(back.shape[2]):
        c[:,:,i] = back[:,:,i] * (1-alpha)
    return c


if __name__ == '__main__':
    window = np.array(cv2.imread('window.jpg')) / 255.0
    back1 = np.array(cv2.imread('./pic1/flowers-backA.jpg')) / 255.0
    back2 = np.array(cv2.imread('./pic1/flowers-backB.jpg')) / 255.0
    comp1 = np.array(cv2.imread('./pic1/flowers-compA.jpg')) / 255.0
    comp2 = np.array(cv2.imread('./pic1/flowers-compB.jpg')) / 255.0
    fg, alpha = matting(back1, back2, comp1, comp2)
    cv2.imwrite('./pic1/flowers-alpha.jpg', alpha*255.0)
    cv2.imwrite('./pic1/flowers-foreground.jpg', fg*255.0)
    b = multiply_alpha(alpha, window)
    composite = np.clip(fg + b, 0, 1)
    # cv2.imshow('flowers-composite.jpg', composite)
    # cv2.waitKey(0)
    cv2.imwrite('./pic1/flowers-composite.jpg', composite*255.0)

    back1 = np.array(cv2.imread('./pic2/leaves-backA.jpg')) / 255.0
    back2 = np.array(cv2.imread('./pic2/leaves-backB.jpg')) / 255.0
    comp1 = np.array(cv2.imread('./pic2/leaves-compA.jpg')) / 255.0
    comp2 = np.array(cv2.imread('./pic2/leaves-compB.jpg')) / 255.0
    fg, alpha = matting(back1, back2, comp1, comp2)
    cv2.imwrite('./pic2/leaves-alpha.jpg', alpha*255.0)
    cv2.imwrite('./pic2/leaves-foreground.jpg', fg*255.0)
    b = multiply_alpha(alpha, window)
    composite = np.clip(fg + b, 0, 1)
    # cv2.imshow('leaves-composite.jpg', composite)
    # cv2.waitKey(0)
    cv2.imwrite('./pic2/leaves-composite.jpg', composite*255.0)