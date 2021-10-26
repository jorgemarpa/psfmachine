import socket
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_3d
import astropy.units as u
from astropy.table import Table

pd.set_option("mode.chained_assignment", None)


def query_local_db(ras, decs, rads, magnitude_limit=18):
    """
    Does a cone search in a local copy of Gaia EDR3 of Kepler Field in case there's no
    internet connection, e.g. when running at NAS because computing nodes do not have
    internet access.
    """
    # log.info("Using local table")
    if socket.gethostname() != "NASAs-MacBook-Pro.local":
        path = "/nobackup/jimartin/ADAP/data/catalogs"
    else:
        path = "/Users/jorgemarpa/Work/BAERI/ADAP/data/catalogs"

    # open table
    table = Table.read(f"{path}/kic_x_gaia_edr3_kepler_field.fits").to_pandas()

    # filter sources around TPFs, square approx
    ralim = [np.min(ras) - 0.1, np.max(ras) + 0.1]
    declim = [np.min(decs) - 0.1, np.max(decs) + 0.1]
    table = table.query(
        f"ra > {ralim[0]} and ra < {ralim[1]} and "
        f"dec > {declim[0]} and dec < {declim[1]} and "
        f"phot_g_mean_mag <= {magnitude_limit}"
    )
    # clean table
    table.loc[:, "designation"] = table["designation"].str.decode("utf-8")
    table.loc[:, "kic"] = table["kic"].fillna(0)
    catalog = SkyCoord(
        ra=table["ra"], dec=table["dec"], frame="icrs", unit=(u.deg, u.deg)
    )

    results = []
    for ra, dec, rad in zip(ras, decs, rads):
        c = SkyCoord(ra=[ra], dec=[dec], frame="icrs", unit=(u.deg, u.deg))
        idxc, idxcatalog, d2d, _ = catalog.search_around_sky(c, rad * u.deg)
        results.append(table.iloc[idxcatalog])

    return (
        pd.concat(results, axis=0)
        .drop_duplicates(subset=["designation"])
        .sort_values("designation")
        .reset_index(drop=True)
    )
