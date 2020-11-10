import os, math

import numpy as np
import matplotlib.pyplot as plt

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else:
                tmp_norm = (tmp - tmp.min()) / (tmp.max() - tmp.min())
                canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = tmp_norm
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas

def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def random_noise(batch_size, zdim):

    return np.random.uniform(-1, 1, [batch_size, zdim]).astype(np.float32)

def training(neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    make_dir(path="training")
    result_list = ["from_noise"]
    if(neuralnet.zdim == 2): result_list.append("latent_walk")
    for result_name in result_list: make_dir(path=os.path.join("training", result_name))

    iteration = 0
    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        x_tr, y_tr, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch
        z_tr = random_noise(x_tr.shape[0], neuralnet.zdim)
        step_dict = neuralnet.step(x=x_tr, y=y_tr, z=z_tr, training=False)
        x_fake = step_dict['x_fake']
        plt.imsave(os.path.join("training", "from_noise", "%08d.png" %(epoch)), dat2canvas(data=x_fake))

        if(neuralnet.zdim == 2):
            x_values = np.linspace(-3, 3, test_sq)
            y_values = np.linspace(-3, 3, test_sq)
            z_latents = None
            for y_loc, y_val in enumerate(y_values):
                for x_loc, x_val in enumerate(x_values):
                    z_latent = np.reshape(np.array([y_val, x_val]), (1, neuralnet.zdim))
                    if(z_latents is None): z_latents = z_latent
                    else: z_latents = np.append(z_latents, z_latent, axis=0)
            step_dict = neuralnet.step(x=x_tr, y=y_tr, z=z_latents, training=False)
            x_fake = step_dict['x_fake']
            z_tr = random_noise(x_tr.shape[0], neuralnet.zdim)
            plt.imsave(os.path.join("training", "latent_walk", "%08d.png" %(epoch)), dat2canvas(data=x_fake))


        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size)
            z_tr = random_noise(x_tr.shape[0], neuralnet.zdim)
            step_dict = neuralnet.step(x=x_tr, y=y_tr, z=z_tr, iteration=iteration, training=True)

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration)  D:%.3f, G:%.3f" \
            %(epoch, epochs, iteration, step_dict['loss_d'], step_dict['loss_g']))
        neuralnet.save_parameter(model='model_checker', epoch=epoch)

def test(neuralnet, dataset, batch_size):

    print("\nTest...")
    neuralnet.load_parameter(model='model_checker')

    make_dir(path="test")

    test_sq = 20
    test_size = test_sq**2

    for i in range(10):
        x_te, y_te, _ = dataset.next_test(batch_size=test_size)
        z_te = random_noise(test_size, neuralnet.zdim)
        step_dict = neuralnet.step(x=x_te, y=y_te, z=z_te, training=False)
        x_fake = step_dict['x_fake']
        plt.imsave(os.path.join("test", "%08d.png" %(i)), dat2canvas(data=x_fake))
