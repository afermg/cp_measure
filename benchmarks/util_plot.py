from matplotlib.offsetbox import AnchoredText


def generate_label(content, **kwargs):
    text_box = AnchoredText(
        content,
        frameon=False,
        loc=2,
        prop=dict(fontweight="bold", fontsize=15),
        # transform=ax.transAxes,
        borderpad=0,
        **kwargs,
    )
    return text_box
