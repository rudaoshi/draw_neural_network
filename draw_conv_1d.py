


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

sns.set(style="white")

def add_tensor_2d(ax, length = 10, feature_num = 5, notification_box = None, x_label = None, y_label = None):

    # # Generate a large random dataset
    # rs = np.random.RandomState(33)
    # d = pd.DataFrame(data=rs.normal(size=(length, feature_num)),)
    #
    # # Generate a mask for the upper triangle
    # mask = np.zeros_like(d, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True

    uniform_data = np.random.randn(feature_num, length)
    # Draw the heatmap with the mask and correct aspect ratio

    sns.heatmap(uniform_data, xticklabels = False, yticklabels = False, square=True,
                linewidths=.5, ax = ax, cbar = False)

    ax.add_patch(Rectangle((0,0), length, feature_num, fill=False, color="black", linewidth=2))

    if not notification_box:
        notification_box = []

    for box in notification_box:
        ax.add_patch(Rectangle(box["box"][:2], box["box"][2], box["box"][3], fill=True, color=box["color"], alpha=box["alpha"], linewidth=5))


def add_tensor_3d(gs, layer_num, length = 10, feature_num = 5, shift = (0.1, 0.1), notification_box = None, x_label = None, y_label = None):

    ax_box = gs.get_position(plt.gcf())

    pos = [ax_box.x0, ax_box.y0, ax_box.width - (layer_num-1)*shift[0], ax_box.height - (layer_num-1)*shift[1]]

    if not notification_box:
        notification_box = []
    else:
        assert len(notification_box) == layer_num, "size of notification box must be equal to the number of layer"

    axes = []
    for i in range(layer_num):

        cur_ax = plt.gcf().add_axes(pos)

        axes.append(cur_ax)
        if notification_box:
            cur_boxes = notification_box[i]
        else:
            cur_boxes = None

        add_tensor_2d(cur_ax, length, feature_num, x_label = x_label, y_label = y_label, notification_box=cur_boxes)

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

    # inv = plt.gcf().transFigure.inverted()
    #
    # source_disp_coord = source_ax.transData.transform(source_data_coord)
    # print source_disp_coord
    # source_fig_coord = inv.transform(source_disp_coord)
    # print source_fig_coord
    #
    # target_disp_coord = target_ax.transData.transform(target_data_coord)
    # print target_disp_coord
    # target_fig_coord = inv.transform(target_disp_coord)
    # print target_fig_coord
    #
    # line = matplotlib.lines.Line2D((source_fig_coord[0], target_fig_coord[0]), (source_fig_coord[1], target_fig_coord[1]),
    #                                transform=plt.gcf().transFigure)
    #
    # plt.gcf().lines = line,
#    plt.gcf().line(source_fig_coord[0], source_fig_coord[1],
#                    target_fig_coord[0], target_fig_coord[1], head_width=0.05, head_length=0.1, fc='k', ec='k')

#    plt.gcf().line(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')

def add_mapping(patches, colors, start_ratio, patch_size, ind_bgn,
                top_left_list, loc_diff_list, num_show_list, size_list):

    start_loc = top_left_list[ind_bgn] \
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
        + np.array([start_ratio[0] * size_list[ind_bgn],
                    -start_ratio[1] * size_list[ind_bgn]])

    end_loc = top_left_list[ind_bgn + 1] \
        + (num_show_list[ind_bgn + 1] - 1) \
        * np.array(loc_diff_list[ind_bgn + 1]) \
        + np.array([(start_ratio[0] + .5 * patch_size / size_list[ind_bgn]) *
                    size_list[ind_bgn + 1],
                    -(start_ratio[1] - .5 * patch_size / size_list[ind_bgn]) *
                    size_list[ind_bgn + 1]])

    patches.append(Rectangle(start_loc, patch_size, patch_size))
    colors.append(Dark)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Black)
    patches.append(Line2D([start_loc[0] + patch_size, end_loc[0]],
                          [start_loc[1], end_loc[1]]))
    colors.append(Black)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1] + patch_size, end_loc[1]]))
    colors.append(Black)
    patches.append(Line2D([start_loc[0] + patch_size, end_loc[0]],
                          [start_loc[1] + patch_size, end_loc[1]]))
    colors.append(Black)


def label(xy, text, xy_off=[0, 4]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=8)


def draw():
    fc_unit_size = 2
    layer_width = 40

    patches = []
    colors = []

    fig, ax = plt.subplots()


    ############################
    # conv layers
    size_list = [32, 18, 10, 6, 4]
    num_list = [3, 32, 32, 48, 48]
    x_diff_list = [0, layer_width, layer_width, layer_width, layer_width]
    text_list = ['Inputs'] + ['Feature\nmaps'] * (len(size_list) - 1)
    loc_diff_list = [[3, -3]] * len(size_list)

    num_show_list = map(min, num_list, [NumConvMax] * len(num_list))
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

    for ind in range(len(size_list)):
        position = [ind * 100, 0, 100, 100]
        add_tensor_2d(poistion=position)
        label(top_left_list[ind], text_list[ind] + '\n{}@{}x{}'.format(
            num_list[ind], size_list[ind], size_list[ind]))


    ############################
    # in between layers
    start_ratio_list = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0.4, 0.8]]
    patch_size_list = [5, 2, 5, 2]
    ind_bgn_list = range(len(patch_size_list))
    text_list = ['Convolution', 'Max-pooling', 'Convolution', 'Max-pooling']

    for ind in range(len(patch_size_list)):
        add_mapping(patches, colors, start_ratio_list[ind],
                    patch_size_list[ind], ind,
                    top_left_list, loc_diff_list, num_show_list, size_list)
        label(top_left_list[ind], text_list[ind] + '\n{}x{} kernel'.format(
            patch_size_list[ind], patch_size_list[ind]), xy_off=[26, -65])


    ############################
    # fully connected layers
    size_list = [fc_unit_size, fc_unit_size, fc_unit_size]
    num_list = [768, 500, 2]
    num_show_list = map(min, num_list, [NumFcMax] * len(num_list))
    x_diff_list = [sum(x_diff_list) + layer_width, layer_width, layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Hidden\nunits'] * (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        position = [1000+ind * 100, 0, 100, 100]
        add_tensor_2d(position)
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]))

    text_list = ['Flatten\n', 'Fully\nconnected', 'Fully\nconnected']

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[-10, -65])

    ############################
    colors += [0, 1]
    collection = PatchCollection(patches, cmap=plt.cm.gray)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
#    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    plt.show()
#    fig.set_size_inches(8, 2.5)

    fig_dir = './'
    fig_ext = '.png'
    fig.savefig(os.path.join(fig_dir, 'convnet_fig' + fig_ext),
                bbox_inches='tight', pad_inches=0)


import matplotlib.gridspec as gridspec

if __name__ == "__main__":


    text_length = 20
    embedding_dim = 5
    conv_dim = 10

    embedding_layer_pos = [0, 0, 1, 1]
    #
    # add_layer(ax, embedding_layer_pos, length=text_length, feature_num=embedding_dim)
    #
    # fig_dir = './'
    # fig_ext = '.png'
    # f.savefig(os.path.join(fig_dir, 'embedding_layer' + fig_ext),
    #             bbox_inches='tight', pad_inches=0)
    gs = gridspec.GridSpec(4, 1)

    embedding_ax = plt.subplot(gs[3, :])
    note_boxes = [{"box": [1.5,0, 3, embedding_dim], "color":"blue", "alpha" : 0.6},
                  {"box": [7.5,0, 3, embedding_dim], "color":"red", "alpha" : 0.6},
                  {"box": [16.5, 0, 3, embedding_dim], "color": "green", "alpha": 0.6}]
    add_tensor_2d(embedding_ax, length=text_length, feature_num=embedding_dim, x_label="words", y_label= "embedding", notification_box=note_boxes)


#    conv_ax = plt.subplot(gs[1:3, :])

    shift = [0.1,0.025]

    note_boxes = [[{"box": [1, 0, 1, conv_dim], "color": "blue", "alpha": 0.6}, {"box": [0, 8, text_length, 1], "color": "cyan", "alpha": 0.6}, ],
                  [{"box": [4, 0, 1, conv_dim], "color": "red", "alpha": 0.6}, {"box": [0, 5, text_length, 1], "color": "magenta", "alpha": 0.6},],
                  [{"box": [16, 0, 1, conv_dim], "color": "green", "alpha": 0.6}, {"box": [0, 2, text_length, 1], "color": "orange", "alpha": 0.6},]]
    conv_axes = add_tensor_3d(gs[1:3, :], 3, length=text_length, feature_num=conv_dim, shift = shift, notification_box=note_boxes)



#    add_connection(embedding_ax, (3, embedding_dim), conv_axes[0] , (1.5, 0))
#    add_connection(embedding_ax, (9, embedding_dim), conv_axes[1], (4.5, 0))
#    add_connection(embedding_ax, (18, embedding_dim), conv_axes[2], (16.5, 0))

    max_pool_ax = plt.subplot(gs[0, :])
    note_boxes = [{"box": [5, 0, 1, 1], "color": "cyan", "alpha": 0.6},
                  {"box": [16, 0, 1, 1], "color": "magenta", "alpha": 0.6},
                  {"box": [35, 0, 1, 1], "color": "orange", "alpha": 0.6}]
    add_tensor_2d(max_pool_ax, length=text_length*2, feature_num=1, notification_box=note_boxes)


#    add_connection(conv_axes[0], (0, 8.5), max_pool_ax , (5.5, 0))
#    add_connection(conv_axes[1], (0, 5.5), max_pool_ax, (16.5, 0))
#    add_connection(conv_axes[2], (text_length, 2.5), max_pool_ax, (35.5, 0))


    plt.gcf().tight_layout()
    plt.show()

    # fig_dir = './'
    # fig_ext = '.png'
    # plt.gcf().savefig(os.path.join(fig_dir, 'language_model.'+ fig_ext),
    #            pad_inches=0)
