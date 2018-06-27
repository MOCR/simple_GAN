
import NetBluePrint as nbp
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc

import os

from tensorflow.python.client import timeline


sess = tf.Session()
with sess.as_default():
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    wk, cl = nbp.create_workflow(None, [["load_dataset", {"batchsize" : 128, "dataset" : "celebA" , "resize_dim" : [128,128], "central_crop" : True}],
                                        ["input_layer", {"new_input": [128,1,1,128]}],
                                        ["tf.random_normal"],
                                        ["network", ["generator", "generator_128_128"]],
                                        ["bridge_layer", {"bridge_name" : "generated_images"}],
                                        ["network", ["discriminator", "discriminator_128_128"]],
                                        ["bridge_layer", {"bridge_name": "discr_fake"}],
                                        ["set_adam_optimizer", {"learning_rate" : 0.0002, "beta" : 0.5}],
                                        ["softmax_loss", {"labels": 0.0}],
                                        ["compute_gradients", {"scopes" : ["discriminator"], "clear_losses" : True}],
                                        ["trainer"],
                                        ["save_in_list", {"list_name" : "trainers"}],
                                        ["clear_field", {"field_name" : "gradients"}],
                                        ["input_layer", {"new_input": "@:/dataset/image"}],
                                        ["network", ["discriminator"]],
                                        ["bridge_layer", {"bridge_name": "discr_true"}],
                                        ["softmax_loss", {"labels" : 1.0}],
                                        ["compute_gradients", {"scopes": ["discriminator"], "clear_losses": True}],
                                        ["trainer"],
                                        ["save_in_list", {"list_name": "trainers"}],
                                        ["clear_field", {"field_name": "gradients"}],
                                        ["set_adam_optimizer", {"learning_rate" : 0.0002, "beta" : 0.5}],
                                        ["input_layer", {"new_input": "@:/bridges/discr_fake"}],
                                        ["softmax_loss", {"labels": 1.0}],
                                        ["compute_gradients", {"scopes": ["generator"], "clear_losses": True}],
                                        ["trainer"],
                                        ["save_in_list", {"list_name": "trainers"}],
                                        ["clear_field", {"field_name": "gradients"}],
                                        ["initializer"]], "basic_gan_training")
    #plt.ion()
    for i in range(100000):
        if i%250==0:
            sess.run(cl["trainers"][i%len(cl["trainers"])], options=options, run_metadata=run_metadata)
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('timeline_'+str(i)+'.json', 'w') as f:
                f.write(chrome_trace)
        else:
            sess.run(cl["trainers"][i%len(cl["trainers"])])
        if i%1000==0:
            print(i)
            img, res = sess.run([cl["dataset"]["image"], cl["bridges"]["generated_images"]])
            scipy.misc.imsave('generated.jpg', res[0])
            #scipy.misc.toimage(res, cmin=0.0, cmax=1.0).save('generated.jpg')
            os.system('tiv -w 128 generated.jpg')
            #plt.cla()
            #plt.imshow(np.vstack((img[0], res[0])))
            #plt.show()
    plt.show()

                          # ["conv_relu", {"out_size" : 16, "kernel_size" : 3}]], "test")

