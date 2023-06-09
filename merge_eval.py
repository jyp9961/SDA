import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

if __name__ == '__main__':
    directory = 'results'
    plot_type = 'AU7005'
    directory = os.path.join(directory, plot_type)
    env_names = [['ball_in_cup', 'catch'], ['cartpole', 'swingup'], ['finger', 'spin'], ['walker', 'stand'], ['walker', 'walk']]
    
    fnames = []
    for env_name in env_names:
        domain_name, task_name = env_name
        fname = os.path.join(directory, '{}_{}_gb.png'.format(domain_name, task_name))
        fnames.append(fname)

    num_x = 5
    num_y = 1
    x_start, x_end = 0, 1000
    y_start, y_end = 0, 2000
    b = np.ones(((x_end-x_start)*num_x,(y_end-y_start)*num_y,3), dtype=np.uint8)*255
    i = 0
    for fname in fnames:
        a = cv2.imread(fname)
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        #plt.imshow(a)
        #plt.show()
        #a = a[x_start:x_end, y_start:y_end,:]
        print(np.shape(a))
        x,y = int(i/num_y), int(i%num_y)
        print(x,y)
        np.copyto(b[x*(x_end-x_start):(x+1)*(x_end-x_start), y*(y_end-y_start):(y+1)*(y_end-y_start), :],a)
        print(np.shape(a), np.max(a), np.min(a))
        i += 1
    
    print(np.shape(b), np.max(b), np.min(b))
    b = Image.fromarray(b)
    b.save(os.path.join(directory,'total_gb.png'))       