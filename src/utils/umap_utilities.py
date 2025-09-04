import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from bokeh.palettes import Turbo256  # A palette with 20 distinct colors
from numpy.typing import NDArray
from torch import Tensor
from umap import UMAP

type Arraylike = NDArray | Tensor


def perform_umap(input_data: Arraylike, genes: list[str], **umap_kwargs):
    """
    Perform UMAP on the input data and return a dataframe of results

    Args:
        input_data (Arraylike): Input data
        genes (list[str]): metadata about genes
        **umap_kwargs (Dict): Additional inputs to `umap.UMAP`

    Returns:
        pl.DataFrame (Dataframe)
    """
    reducer = UMAP(**umap_kwargs)
    reduced = reducer.fit_transform(input_data)

    embedding_df = pl.DataFrame(reduced, schema=["UMAP-0", "UMAP-1"]).with_columns(
        pl.Series("gene", genes)
    )

    return embedding_df


def plot_output_seaborn(embedding_df: pl.DataFrame, figsize: tuple[int, int] = (9, 8)):
    """
    Plot UMAP results using seaborn
    """
    fig, ax = plt.subplots(figsize=figsize)
    avg_data = embedding_df.group_by("gene").agg(pl.all().median())

    sns.scatterplot(
        x=embedding_df["UMAP-0"],
        y=embedding_df["UMAP-1"],
        hue="gene",
        ax=ax,
        alpha=0.7,
    )
    handles, labels = plt.gca().get_legend_handles_labels()

    sorted_pairs = sorted(zip(labels, handles))
    sorted_labels = [label for label, handle in sorted_pairs]
    sorted_handles = [handle for label, handle in sorted_pairs]
    plt.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1, 1), ncol=3)

    for row in avg_data.iter_rows(named=True):
        ax.text(
            x=row["UMAP-0"], y=row["UMAP-1"], s=row["gene"], ha="center", va="center"
        )

    ax.set_title("Projection UMAP", size=12)

    return fig


def plot_output_bokeh(embedding_df: pl.DataFrame):
    """
    Plot UMAP results using bokeh
    """

    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure
    from bokeh.transform import factor_cmap

    embedding_pd_df = embedding_df.to_pandas()
    source = ColumnDataSource(embedding_pd_df)

    spread_palette = [
        Turbo256[i * 5] for i, _ in enumerate(embedding_df["gene"].unique())
    ]

    # --- 3. Define the Hover Tool ---
    # The tooltip shows the 'Gene' and its (x,y) coordinates
    # @gene refers to the 'gene' column in the ColumnDataSource
    hover = HoverTool(
        tooltips=[
            ("Gene", "@gene"),
            ("(x,y)", "($x, $y)"),
        ]
    )

    # --- 4. Create the Plot ---
    p = figure(
        width=800,
        height=600,
        title="Projection UMAP",
        tools=[
            hover,
            "pan,wheel_zoom,box_zoom,reset,save",
        ],  # Add hover to the default tools
    )

    # --- 5. Add the Scatter Glyph ---
    # Use factor_cmap to color points by the 'gene' column
    unique_genes = sorted(embedding_pd_df["gene"].unique())
    p.scatter(
        x="UMAP-0",
        y="UMAP-1",
        source=source,
        legend_group="gene",  # Group points by gene for the legend
        alpha=0.7,
        size=7,
        color=factor_cmap("gene", palette=spread_palette, factors=unique_genes),
    )

    # --- 6. Customize the Plot ---
    p.legend.title = "Genes"
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"  # Click legend items to hide genes!
    p.legend.ncols = 3
    p.add_layout(p.legend[0], "right")  # Move legend outside the plot area

    return p


def plot_output_plotly(embedding_df: pl.DataFrame):
    """
    Plot UMAP results using plotly
    """
    import plotly.express as px
    import plotly.graph_objects as go

    # Plotly Express works best with Pandas DataFrames
    avg_data = embedding_df.group_by("gene").agg(pl.all().median()).sort("gene")
    spread_palette = [
        Turbo256[i * 5] for i, _ in enumerate(embedding_df["gene"].unique())
    ]

    spread_palette = np.random.choice(spread_palette, size=len(spread_palette))

    # Convert to Pandas for Plotly
    embedding_pd_df = embedding_df.to_pandas()
    avg_data_pd = avg_data.to_pandas()

    # --- 3. Create the Base Scatter Plot ---
    # We use Plotly Express for the main scatter plot as it's great at handling colors and legends
    fig = go.Figure()

    fig = px.scatter(
        embedding_pd_df,
        x="UMAP-0",
        y="UMAP-1",
        color="gene",
        color_discrete_sequence=spread_palette,
        # We remove the default hover info from individual points to focus on the centroid labels
        hover_data={},
        category_orders={"gene": np.sort(embedding_pd_df["gene"].unique())},
    )

    # --- 4. Add the Centroid Labels as a New Layer (Trace) ---
    fig.add_trace(
        go.Scatter(
            x=avg_data_pd["UMAP-0"],
            y=avg_data_pd["UMAP-1"],
            mode="text",  # This is the key: specifies that we are plotting text, not markers
            text=avg_data_pd["gene"],
            textfont=dict(color="black", size=12),
            hoverinfo="none",  # Disable hover for the text labels themselves
            showlegend=False,  # Hide this trace from the legend
        )
    )

    # --- 5. Update Layout for Consistent Sizing and Appearance ---
    # Your original figsize=(9, 8) translates to roughly width=900, height=800
    fig.update_layout(
        title="Projection UMAP",
        width=900,  # Set figure width in pixels
        height=800,  # Set figure height in pixels
        legend_title_text="Gene",
    )

    # Improve marker appearance
    fig.update_traces(marker=dict(size=6, opacity=0.6))

    return fig
