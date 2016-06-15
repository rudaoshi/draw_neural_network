


import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.collections import PatchCollection


NumConvMax = 8
NumFcMax = 20
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.3
Black = 0.


from string import letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms

sns.set(style="white")


def calculate_region(box, width, height, cel_size = 0.01):

    if isinstance(box, list):
        center = (box[0] + box[2]/2, box[1], box[3]/2)
    elif isinstance(box, matplotlib.transforms.Bbox):
        center = (box.x0 + box.width/2, box.y0, box.height/2)
    else:
        raise Exception("unknown box type")

    region = (center[0] - width/2.0*cel_size, center[1] - height/2.0 * cel_size, width * cel_size, height*cel_size)

    return region


def add_tensor_2d(box, length = 10, feature_num = 5, notification_box = None, x_label = None, y_label = None):

    # # Generate a large random dataset
    # rs = np.random.RandomState(33)
    # d = pd.DataFrame(data=rs.normal(size=(length, feature_num)),)
    #
    # # Generate a mask for the upper triangle
    # mask = np.zeros_like(d, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True

    uniform_data = np.random.randn(feature_num, length)
    # Draw the heatmap with the mask and correct aspect ratio

    region = calculate_region(box, length, feature_num)
    cur_ax = plt.gcf().add_axes(region)

    sns.heatmap(uniform_data, xticklabels = False, yticklabels = False, square=True,
                linewidths=.5, ax = cur_ax, cbar = False)

    cur_ax.add_patch(Rectangle((0,0), length, feature_num, fill=False, color="black", linewidth=2))

    if not notification_box:
        notification_box = []

    for box in notification_box:
        cur_ax.add_patch(Rectangle(box["box"][:2], box["box"][2], box["box"][3], fill=True, color=box["color"], alpha=box["alpha"], linewidth=5))

    return cur_ax

import copy
def add_tensor_3d(box, layer_num, length = 10, feature_num = 5, shift = (0.1, 0.1), notification_box = None, x_label = None, y_label = None):


    pos = [box.x0, box.y0, box.width , box.height ]

    if not notification_box:
        notification_box = []
    else:
        assert len(notification_box) == layer_num, "size of notification box must be equal to the number of layer"

    axes = []
    for i in range(layer_num):
        print pos



        if notification_box:
            cur_boxes = notification_box[i]
        else:
            cur_boxes = None


        cur_ax = add_tensor_2d(pos, length, feature_num, x_label = x_label, y_label = y_label, notification_box=cur_boxes)
        axes.append(cur_ax)
        x_label = None
        y_label = None

        plt.hold(True)

        pos[0] += shift[0]
        pos[1] += shift[1]

    return axes


def add_connection(source_ax, source_data_coord,  target_ax, target_data_coord):
    con = ConnectionPatch(xyA=source_data_coord, xyB=target_data_coord,
                          coordsA='data', coordsB='data',
                          axesA=source_ax, axesB=target_ax,
                          arrowstyle='->', clip_on=False, linewidth=3)
    con.set_zorder(10)

    source_ax.add_artist(con)



import matplotlib.gridspec as gridspec



if __name__ == "__main__":


    text_length = 20
    embedding_dim = 5
    conv_dim = 10

    feature_component_dim = 10

    embedding_layer_pos = [0, 0, 1, 1]
    #
    # add_layer(ax, embedding_layer_pos, length=text_length, feature_num=embedding_dim)
    #
    # fig_dir = './'
    # fig_ext = '.png'
    # f.savefig(os.path.join(fig_dir, 'embedding_layer' + fig_ext),
    #             bbox_inches='tight', pad_inches=0)

    plt.gcf().set_size_inches(16, 10)
    gs = gridspec.GridSpec(3, 7)

    #embedding_ax = plt.subplot(gs[3, 0])
    note_boxes = [{"box": [0.5,0, 2, embedding_dim], "color":"blue", "alpha" : 0.6},
                  {"box": [9.5,0, 2, embedding_dim], "color":"red", "alpha" : 0.6},
                  {"box": [16.5, 0, 2, embedding_dim], "color": "yellow", "alpha": 0.6}]
    embedding_ax = add_tensor_2d(gs[2, 0].get_position(plt.gcf()), length=text_length, feature_num=embedding_dim, x_label="words", y_label= "embedding", notification_box=note_boxes)


#    conv_ax = plt.subplot(gs[1:3, :])

    shift = [0.025,0.025]

    note_boxes = [[{"box": [1, 0, 1, conv_dim], "color": "blue", "alpha": 0.6}, {"box": [0, 16, text_length, 1], "color": "cyan", "alpha": 0.6}, ],
                  [{"box": [4, 0, 1, conv_dim], "color": "red", "alpha": 0.6}, {"box": [0, 12, text_length, 1], "color": "magenta", "alpha": 0.6},],
                  [{"box": [16, 0, 1, conv_dim], "color": "yellow", "alpha": 0.6}, {"box": [0, 3, text_length, 1],
                                                                                "color": "orange", "alpha": 0.6}]]

    conv_axes = add_tensor_3d(gs[1, 0].get_position(plt.gcf()), 3, length=text_length, feature_num=conv_dim, shift = shift, notification_box=note_boxes)


#    add_connection(embedding_ax, (1.5, embedding_dim), conv_axes[0] , (1.5, 0))
#    add_connection(embedding_ax, (4.5, embedding_dim), conv_axes[0], (1.5, 0))
#    add_connection(embedding_ax, (7.5, embedding_dim), conv_axes[1], (4.5, 0))
#    add_connection(embedding_ax, (10.5, embedding_dim), conv_axes[1], (4.5, 0))


    note_boxes = [{"box": [5, 0, 1, 1], "color": "cyan", "alpha": 0.6},
                  {"box": [16, 0, 1, 1], "color": "magenta", "alpha": 0.6},
                  {"box": [35, 0, 1, 1], "color": "orange", "alpha": 0.6}]
    max_pool_ax = add_tensor_2d(gs[0, 0].get_position(plt.gcf()), length=feature_component_dim, feature_num=1, notification_box=note_boxes)


#    add_connection(conv_axes[0], (0, 8.5), max_pool_ax , (5.5, 0))
#    add_connection(conv_axes[1], (0, 5.5), max_pool_ax, (16.5, 0))

    # embedding_ax = plt.subplot(gs[3, 0])
    note_boxes = [{"box": [0.5, 0, 2, embedding_dim], "color": "blue", "alpha": 0.6},
                  {"box": [9.5, 0, 2, embedding_dim], "color": "red", "alpha": 0.6},
                  {"box": [16.5, 0, 2, embedding_dim], "color": "yellow", "alpha": 0.6}]
    embedding_ax = add_tensor_2d(gs[2, 6].get_position(plt.gcf()), length=text_length, feature_num=embedding_dim,
                                 x_label="words", y_label="embedding", notification_box=note_boxes)

    #    conv_ax = plt.subplot(gs[1:3, :])

    shift = [0.025, 0.025]

    note_boxes = [[{"box": [1, 0, 1, conv_dim], "color": "blue", "alpha": 0.6},
                   {"box": [0, 16, text_length, 1], "color": "cyan", "alpha": 0.6}, ],
                  [{"box": [4, 0, 1, conv_dim], "color": "red", "alpha": 0.6},
                   {"box": [0, 12, text_length, 1], "color": "magenta", "alpha": 0.6}, ],
                  [{"box": [16, 0, 1, conv_dim], "color": "yellow", "alpha": 0.6}, {"box": [0, 3, text_length, 1],
                                                                                    "color": "orange", "alpha": 0.6}]]

    conv_axes = add_tensor_3d(gs[1, 6].get_position(plt.gcf()), 3, length=text_length, feature_num=conv_dim,
                              shift=shift, notification_box=note_boxes)

    #    add_connection(embedding_ax, (1.5, embedding_dim), conv_axes[0] , (1.5, 0))
    #    add_connection(embedding_ax, (4.5, embedding_dim), conv_axes[0], (1.5, 0))
    #    add_connection(embedding_ax, (7.5, embedding_dim), conv_axes[1], (4.5, 0))
    #    add_connection(embedding_ax, (10.5, embedding_dim), conv_axes[1], (4.5, 0))


    note_boxes = [{"box": [5, 0, 1, 1], "color": "cyan", "alpha": 0.6},
                  {"box": [16, 0, 1, 1], "color": "magenta", "alpha": 0.6},
                  {"box": [35, 0, 1, 1], "color": "orange", "alpha": 0.6}]
    max_pool_ax = add_tensor_2d(gs[0, 6].get_position(plt.gcf()), length=feature_component_dim, feature_num=1,
                                notification_box=note_boxes)

    for i in range(1,6):
        add_tensor_2d(gs[0, i].get_position(plt.gcf()), length=feature_component_dim, feature_num=1)


    #    plt.gcf().tight_layout()
    #plt.show()

    fig_dir = './'
    fig_ext = '.png'
    plt.gcf().savefig(os.path.join(fig_dir, 'language_model.'+ fig_ext),
                pad_inches=0)
