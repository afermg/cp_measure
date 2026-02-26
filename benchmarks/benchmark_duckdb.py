#!/usr/bin/env python
"""
Compare Cellprofiler and cp_measure data using DuckDB.
Plot the comparison.
"""

from pathlib import Path
import duckdb
import matplotlib
import matplotlib.pyplot as plt
import pooch
import seaborn as sns
from duckdb.sqltypes import VARCHAR
from pooch import Unzip

from parse_features import get_feature_groups
from util_names import get_cpm_to_measurement_mapper
from util_plot import generate_label

figs_dir = Path("figs")
figs_dir.mkdir(parents=True, exist_ok=True)

profiles_dir = Path("/datastore") / "alan" / "cp_measure" / "profiles_via_masks"
out_dir = Path("/datastore") / "alan" / "cp_measure" / "benchmark_comparison"
out_dir.mkdir(parents=True, exist_ok=True)

cpmeasure_parquet = profiles_dir / "first_set.parquet"
correlation_parquet = profiles_dir / "correlation.parquet"


def trim_features(name: str) -> str:
    """
    Trim object and channel information from CellProfiler
    """
    replacement_ch = [f"_Orig{x}" for x in ("DNA", "AGP", "ER", "Mito", "RNA")]
    replacement_mea = ["AreaShape_", "Texture_"]

    for ch in replacement_ch:
        name = name.replace(ch, "")
    for mea in replacement_mea:
        name = name.replace(mea, "")
    return name


con = duckdb.connect()
con.create_function("trim_features", trim_features, [VARCHAR], VARCHAR)

cp_data = pooch.retrieve(
    "https://zenodo.org/api/records/15505477/files/cellprofiler_analysis.zip/content",
    known_hash="1ca6c08955336d15832fc6dc5c15000990f4dd4733e47a06030a416e7ac7a3e9",
    processor=Unzip(),
)

csv_files = [x for x in cp_data if Path(x).stem.split("_")[-1] in ("Cells", "Nuclei")]
image_table = [x for x in cp_data if x.endswith("Image.csv")][0]

con.sql(
    f"CREATE OR REPLACE TABLE orig_profiles AS (SELECT *, split_part(parse_filename(filename, true),'_',3) AS object FROM read_csv({csv_files}, filename=True, union_by_name=True))"
)

# Consensus: Group by ImageNumber, object -> median
view_info = con.sql("SELECT #1,#2 FROM (DESCRIBE orig_profiles)").fetchall()
numeric_cols = []
for col_name, col_type in view_info:
    if col_type == "DOUBLE":
        numeric_cols.append(col_name)

median_exprs = [f'MEDIAN("{col}") AS "{col}"' for col in numeric_cols]
median_query = f"""
    CREATE OR REPLACE VIEW orig_consensus AS
    SELECT ImageNumber, object, {", ".join(median_exprs)}
    FROM orig_profiles
    GROUP BY ImageNumber, object
"""
con.sql(median_query)


# parse_features.get_feature_groups returns a pyarrow Table
parsed_arrow = get_feature_groups(tuple(numeric_cols), ("feature", "channel", "suffix"))

# Create 'parsed' view
# DuckDB can directly query Python variables that are Arrow tables.
con.sql("""
    CREATE OR REPLACE TABLE parsed_features AS 
    SELECT 
        fullname,
        feature,
        regexp_replace(channel, '^Orig', '') AS channel,
        suffix
    FROM parsed_arrow
""")

# Unpivot consensus
unpivot_query = """
    CREATE OR REPLACE VIEW orig_unpivot AS
    UNPIVOT orig_consensus
    ON COLUMNS(* EXCLUDE (ImageNumber, object))
    INTO NAME fullname VALUE value
"""
con.sql(unpivot_query)

# Join with parsed
con.sql(
    "CREATE OR REPLACE VIEW with_parsed AS SELECT * FROM orig_unpivot NATURAL JOIN parsed_features"
)

# With trimmed
con.sql(
    "CREATE OR REPLACE VIEW with_trimmed AS SELECT *, trim_features(fullname) AS cpm_id FROM with_parsed"
)

# Image metadata
con.sql(
    f"CREATE OR REPLACE VIEW image_table AS SELECT ImageNumber, COLUMNS('FileName.*') FROM '{image_table}'"
)
con.sql(
    """
    CREATE OR REPLACE VIEW imageid_mapper AS (
    SELECT ImageNumber, split_part(name, '_Orig', 2) AS channel,
    split_part(value, '_', 2) AS site,
    split_part(value, '_', 1) AS gene,
    value AS filename
    FROM (UNPIVOT image_table ON COLUMNS('FileName.*')))
    """
)
# CellProfiler Tidy
cellprof_tidy_query = """
    SELECT 
        with_trimmed.* EXCLUDE(value), 
        imageid_mapper.site,
        imageid_mapper.gene,
        imageid_mapper.filename,
        CAST(value AS DOUBLE) as CellProfiler 
    FROM with_trimmed 
    NATURAL JOIN imageid_mapper
"""
con.sql(f"CREATE OR REPLACE VIEW cellprof_tidy AS {cellprof_tidy_query}")


# %% cp_measure
mapper = get_cpm_to_measurement_mapper()

con.sql(f"CREATE OR REPLACE TABLE cp_raw AS SELECT * FROM '{cpmeasure_parquet}'")

meta_cols = ["object", "gene", "channel", "site"]

new_consensus_query = f"""
    CREATE OR REPLACE VIEW new_consensus AS
    SELECT {",".join(meta_cols)}, MEDIAN(COLUMNS(* EXCLUDE({",".join(meta_cols)}))) 
    FROM cp_raw
    GROUP BY {", ".join(meta_cols)}
"""
con.sql(new_consensus_query)

# all_cols_new = raw_numeric_cols
chless_feats = [
    x
    for x in numeric_cols
    if "Intensity" not in x
    and "Zernike" not in x
    and "Granularity" not in x
    and "Difference" not in x
    and "InfoMeas" not in x
    and not x.startswith("RadialDistribution")
    and not x.startswith("Sum")
    and not x.startswith("Entropy")
    and not x.startswith("Contrast")
    and not x.startswith("AngularSecondMoment")
    and not x.startswith("Correlation")
    and not x.startswith("Variance")
]

# Unpivot
unpivot_new_query = f"""
    CREATE OR REPLACE VIEW unpivoted AS
    UNPIVOT new_consensus
    ON COLUMNS(* EXCLUDE ({", ".join(meta_cols)}))
    INTO NAME cpm_id VALUE value
"""
con.sql(unpivot_new_query)

# Handle duplicates (channel-less features obtained for multiple channels)
chless_feats_str = ", ".join([f"'{f}'" for f in chless_feats])
dups_query = f"""
    CREATE OR REPLACE VIEW dups AS
    SELECT 
        upper(substring(object, 1, 1)) || substring(object, 2, strlen(object)) AS object, gene, site, cpm_id, value,
        CASE 
            WHEN cpm_id IN ({chless_feats_str}) THEN ''
            ELSE channel 
        END AS channel
    FROM unpivoted
"""
con.sql(dups_query)

# Unique
con.sql("CREATE OR REPLACE VIEW uniq AS SELECT DISTINCT * FROM dups")

# Combine
merged_query = """
    SELECT 
        uniq.object, uniq.gene, uniq.channel, uniq.site, uniq.cpm_id, 
        uniq.value AS cp_measure,
        cellprof_tidy.filename,
        cellprof_tidy.CellProfiler
    FROM uniq
    NATURAL JOIN cellprof_tidy
    ORDER BY object, gene, channel, site, cpm_id
"""
merged_df = con.sql(merged_query).df()  # Convert to pandas for plotting

duckdb.sql(f"COPY cellprof_tidy TO '{out_dir}/benchmark_table.parquet'")
duckdb.sql(f"COPY merged_df TO '{out_dir}/benchmark_table.parquet'")

# %% Plot
# Using pandas/seaborn as in original

mpd = merged_df
plt.close()
g = sns.FacetGrid(
    mpd,
    col="cpm_id",
    col_wrap=4,
    hue="object",
    sharex=False,
    sharey=False,
    legend_out=False,
    hue_kws={"markers": "channel"},
)
g.map(sns.scatterplot, "CellProfiler", "cp_measure", alpha=0.05)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
plt.tight_layout()
plt.savefig(figs_dir / "grid_cp_vs_cpm.svg")
plt.savefig(figs_dir / "grid_cp_vs_cpm.png")

# %% Linear Regression (R2)
r2_query = """
    SELECT 
        cpm_id,
        regr_r2(value, CellProfiler) as r2
    FROM (
        SELECT * FROM uniq NATURAL JOIN cellprof_tidy
    )
    WHERE value IS NOT NULL AND NOT isnan(value)
    GROUP BY cpm_id
    HAVING count(*) > 1 AND abs(avg(value)) > 1e-12
"""
r2_results = con.execute(r2_query).df()

# Post-processing R2 results for plotting
r2_results["Feature"] = r2_results["cpm_id"]
r2_results["Measurement"] = r2_results["Feature"].replace(mapper)

res = r2_results

# %% Final Plot
plt.close()
font = {"family": "sans-serif", "size": 14}
matplotlib.rc("font", **font)

feats_to_show = [
    "Intensity_MedianIntensity",
    "Area",
    "RadialDistribution_RadialCV_1of4",
]
axd = plt.figure(layout="constrained").subplot_mosaic(
    """
    ABC
    DDD
    """
)

# Plot D (R2 swarmplot)
# Handle potential missing "Granularity" logic if not present
if "Measurement" in res.columns:
    subset_res = res[res["Measurement"] != "Granularity"].sort_values("Measurement")
else:
    subset_res = res

if not subset_res.empty:
    g = sns.swarmplot(
        data=subset_res,
        x="Measurement",
        y="r2",
        ax=axd["D"],
        alpha=0.7,
        palette=sns.color_palette("husl", 8),
        hue="Measurement",
    )

    g.set_ylim(0.925, 1.021)
    g.set_ylabel(r"$R^2$")
    pad = 0.25
    axd["D"].add_artist(generate_label("D", pad=pad))
    g.set_xticklabels(
        g.get_xticklabels(), rotation=15, rotation_mode="anchor", ha="right"
    )

# Plot A, B, C (Scatterplots)
for ax_id, featname in zip("ABC", feats_to_show):
    ax = axd[ax_id]
    # Filter for specific feature
    subset = merged_df[merged_df["cpm_id"] == featname]

    if not subset.empty:
        h = sns.scatterplot(
            data=subset,
            x="CellProfiler",
            y="cp_measure",
            hue="object",
            ax=ax,
            alpha=0.05,
            legend=None if ax_id != "C" else True,
        )

        h.set_yticklabels(
            h.get_yticklabels(), rotation=30, ha="right", rotation_mode="anchor"
        )
        h.set_xticklabels(
            h.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
        )
        if ax_id != "A":
            ax.set_ylabel("")

        display_name = featname.removeprefix("RadialDistribution_")
        display_name = display_name.removeprefix("Intensity_")
        ax.set_title(display_name.replace("_", " "))
        sns.despine()
        ax.add_artist(generate_label(ax_id, pad=pad))

        if ax_id == "C":
            if ax.get_legend():
                for lh in ax.get_legend().legend_handles:
                    lh.set_alpha(1)

                sns.move_legend(
                    ax,
                    loc="lower right",
                    bbox_to_anchor=(1.5, 0),
                    frameon=False,
                    bbox_transform=ax.transAxes,
                    handletextpad=-0.5,
                )

plt.savefig(figs_dir / "jump_r2_examples.svg")
plt.savefig(figs_dir / "jump_r2_examples.png")

print("Processing complete. Figures saved.")
