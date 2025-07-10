import seaborn as sns
import torch
from matplotlib import colors as mplcolors


def get_fancy_cmap():
    colors = sns.color_palette('hls', 21)
    gold = mplcolors.to_rgb('gold')
    colors = [gold] + colors[3:] + colors[:2]
    raw_cmap = mplcolors.LinearSegmentedColormap.from_list('Custom', colors)

    def cmap(values):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        return raw_cmap(values)[:, :3]

    return cmap


def get_fancy_color(num):
    values = torch.linspace(0, 1, num + 1)[1:]
    colors = torch.from_numpy(get_fancy_cmap()(values.cpu().numpy())).float()
    return colors
