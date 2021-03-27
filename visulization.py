import matplotlib.pyplot as plt
from cam import plot_ResNet_CAM

if __name__=='__main__':

    # load your image here
    img = None
    # load your model here
    model = None

    fig, ax = plt.subplots()
    AM = plot_ResNet_CAM(img, ax, model)
    plt.show()