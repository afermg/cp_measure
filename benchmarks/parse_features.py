import re
import pyarrow as pa


def get_feature_groups(
    feature_fullnames: tuple[str],
    feature_names: tuple[str] = ("Feature", "Channel", "Suffix"),
) -> pa.Table:
    """Group features in a consistent manner using a regex.

    Parameters
    ----------
    feature_fullnames : tuple[str]
        Tuple of full names of the features.
    feature_names : tuple[str]
        Tuple of names of the features to be extracted.

    Returns
    -------
    pa.Table
        Arrow Table containing the grouped feature information.
    """
    channels = "|".join(
        (
            "DNA",
            "AGP",
            "RNA",
            "ER",
            "Mito",
            "Image",
        )
    )
    chless_feats = "|".join(
        (
            "AreaShape",
            "Neighbors",
            "Location",
            "Count",
            "Number",
            "Parent",
            "Children",
            "ObjectSkeleton",
            "Threshold",
        )
    )

    std = re.compile(rf"(\S+)_(Orig)?({channels})(_.*)?")
    chless = re.compile(f"({chless_feats})_?([a-zA-Z]+)?(.*)?")
    results = []

    for x in feature_fullnames:
        try:
            # Check for standard match first
            std_match = std.findall(x)
            if std_match:
                match = std_match[0]
                # Match groups: Prefix, Orig(optional), Channel, Suffix
                # Output: Feature, Channel, Suffix
                # Feature = Prefix (match[0])
                # Channel = Orig + Channel (match[1] + match[2])
                # Suffix = Suffix (match[3])
                results.append((match[0], match[1] + match[2], match[3]))
            else:
                # Check for channel-less match
                chless_match = chless.findall(x)
                if chless_match:
                    match = chless_match[0]
                    # Match groups: Feature, FeatureSuffix?, Suffix?
                    # Feature = match[0] + match[1]
                    # Channel = ""
                    # Suffix = match[2]
                    results.append((match[0] + match[1], "", match[2]))
                else:
                    # Fallback
                    results.append((x, "", ""))

        except Exception as e:
            print(f"Error processing {x}: {e}")
            results.append((x, "", ""))

    # Transpose results for columnar storage
    columns_data = list(zip(*results)) if results else [(), (), ()]

    # Create Arrow Table
    data_dict = {name: data for name, data in zip(feature_names, columns_data)}
    data_dict["fullname"] = feature_fullnames

    return pa.Table.from_pydict(data_dict)
