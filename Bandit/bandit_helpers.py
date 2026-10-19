
import logging

from typing import Dict, List, Tuple, Union

logger = logging.getLogger(__name__)


def parse_gage(s: str) -> Tuple:
    """Parse a streamgage key-value pair.

    Parse a streamgage key-value pair, separated by '='; that's the reverse of ShellArgs.
    On the command line (argparse) a declaration will typically look like::
        foo=hello or foo="hello world"

    :param s: Streamgage-segment key,value string
    :returns: Tuple of key and values
    """

    # Adapted from: https://gist.github.com/fralau/061a4f6c13251367ef1d9a9a99fb3e8d
    items = s.split('=')
    key = items[0].strip()  # Remove blanks around keys
    value = ''

    if len(items) > 1:
        value = '='.join(items[1:])
    return key, value


def parse_gages(items: List[str]) -> Dict:
    """Parse a list of key-value pairs and return a dictionary.

    :param items: List of key-value pairs

    :returns: Dictionary with key=streamgage_id and value=nhm_seg
    """

    # Adapted from: https://gist.github.com/fralau/061a4f6c13251367ef1d9a9a99fb3e8d
    d = {}
    if items:
        for item in items:
            key, value = parse_gage(item)
            d[key] = int(value)
    return d


def write_gis(gdf, gis_dir, gis_type, layer):
    if gis_type in ['shp']:
        geo_outfile = gis_dir / f'model_{layer}.{gis_type}'
        gdf.to_file(geo_outfile)
    else:
        geo_outfile = gis_dir / f'model_layers.{gis_type}'
        gdf.to_file(geo_outfile, layer=layer)


def write_streamflow(config,
                     poi_gage_ids: List[str],
                     st_date,
                     en_date,
                     out_path,
                     netcdf: bool = False,
                     addl_gages: Dict | None = None,
                     verbose: bool = False) -> None:
    """Write streamflow observations for a model subset.

    Selects the streamflow source and writes it in the requested format
    (netCDF or PRMS ASCII):

    * When the subset has POIs and ``poi_dir`` names a local cache, a
      :class:`~Bandit.points_of_interest.POI` is used, with an online WDFN
      lookup for any ad-hoc gages supplied on the command line.
    * When the subset has POIs but no local cache, observations are retrieved
      online from the USGS Water Data (WDFN) service.
    * When the subset has no POIs, a single dummy streamgage of missing values
      is written so PRMS still receives a valid runoff file.

    :param config: Loaded Bandit configuration
    :param poi_gage_ids: POI gage IDs in the model subset (may be empty)
    :param st_date: start date for retrieving observations
    :param en_date: end date for retrieving observations
    :param out_path: output path without extension; ``.nc`` is appended for
        netCDF output
    :param netcdf: write netCDF instead of PRMS ASCII when True
    :param addl_gages: ad-hoc gages supplied on the command line, if any
    :param verbose: output additional information
    """

    from pathlib import Path

    from Bandit.points_of_interest import POI   # type: ignore
    from Bandit.WDFN import WDFNStreamflow

    out_path = Path(out_path)
    wdfn_api_key = config.wdfn_api_key if config.exists('wdfn_api_key') else None

    # The source is either a local-cache POI or an online WDFN retrieval; both
    # expose write_ascii()/write_netcdf().
    streamflow: Union[POI, WDFNStreamflow]

    if len(poi_gage_ids) > 0:
        # poi_dir may be absent, an empty string, or a bare YAML key (parsed as
        # None); treat all of those as "no local cache".
        poi_dir = config.poi_dir if config.exists('poi_dir') else None
        has_poi_dir = poi_dir is not None and str(poi_dir).strip() != ''

        if has_poi_dir:
            logger.info('Retrieving POIs from local HYDAT and NWIS cache files')

            # When ad-hoc gages were supplied on the command line, enable an
            # online WDFN lookup for any gage that has no observations in the
            # cached POI source files. Gages that cannot be retrieved fall back
            # to NaN entries.
            streamflow = POI(src_path=poi_dir, st_date=st_date, en_date=en_date,
                             gage_ids=poi_gage_ids, online_lookup=bool(addl_gages),
                             api_key=wdfn_api_key, verbose=verbose)
        else:
            # No local POI cache: retrieve observations online from the USGS
            # Water Data (WDFN) service. This replaces the retired NWIS web
            # service (waterservices.usgs.gov).
            logger.info('No poi_dir: retrieving POIs online from the USGS WDFN service.')
            streamflow = WDFNStreamflow(gage_ids=poi_gage_ids, st_date=st_date, en_date=en_date,
                                        api_key=wdfn_api_key, verbose=verbose)
    else:
        # No POIs in the model subset. PRMS still expects a runoff file, so
        # write a single dummy streamgage of missing values.
        logger.info('No POIs exist in model subset; dummy data written.')
        streamflow = WDFNStreamflow(gage_ids=None, st_date=st_date, en_date=en_date,
                                    api_key=wdfn_api_key, verbose=verbose)

    if netcdf:
        streamflow.write_netcdf(filename=out_path.with_suffix('.nc'))
    else:
        streamflow.write_ascii(filename=out_path)
