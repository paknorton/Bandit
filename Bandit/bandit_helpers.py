
import datetime
import logging
import networkx as nx
import re

from collections import OrderedDict

try:
    from typing import Optional, Union, Dict, List, OrderedDict as OrderedDictType
except ImportError:
    # pre-python 3.7.2
    from typing import Optional, Union, Dict, List, MutableMapping as OrderedDictType


bandit_helper_log = logging.getLogger('bandit.helper')


def parse_gage(s):
    """Parse a streamgage key-value pair.

    Parse a streamgage key-value pair, separated by '='; that's the reverse of ShellArgs.
    On the command line (argparse) a declaration will typically look like::
        foo=hello or foo="hello world"

    :param s: str
    :rtype: tuple(key, value)
    """

    # Adapted from: https://gist.github.com/fralau/061a4f6c13251367ef1d9a9a99fb3e8d
    items = s.split('=')
    key = items[0].strip()  # we remove blanks around keys, as is logical
    value = ''

    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
    return key, value


def parse_gages(items):
    """Parse a list of key-value pairs and return a dictionary.

    :param list[str] items: list of key-value pairs

    :returns: key-value dictionary
    :rtype: dict[str, str]
    """

    # Adapted from: https://gist.github.com/fralau/061a4f6c13251367ef1d9a9a99fb3e8d
    d = {}
    if items:
        for item in items:
            key, value = parse_gage(item)
            d[key] = int(value)
    return d


def set_date(adate):
    """Return datetime object given a datetime or string of format YYYY-MM-DD"""
    if isinstance(adate, datetime.date):
        return adate
    else:
        return datetime.datetime(*[int(x) for x in re.split('[- :]', adate)])


def subset_stream_network(dag_ds: nx.classes.digraph.DiGraph,
                          uscutoff_seg: List[int],
                          dsmost_seg: List[int]) -> nx.classes.digraph.DiGraph:
    # Create the upstream graph
    # TODO: 2021-12-01 PAN - the reverse function is pretty inefficient for multi-location
    #       jobs.
    dag_us = dag_ds.reverse()
    bandit_helper_log.debug('Number of NHM upstream nodes: {}'.format(dag_us.number_of_nodes()))
    bandit_helper_log.debug('Number of NHM upstream edges: {}'.format(dag_us.number_of_edges()))

    # Trim the u/s graph to remove segments above the u/s cutoff segments
    try:
        for xx in uscutoff_seg:
            try:
                dag_us.remove_nodes_from(nx.dfs_predecessors(dag_us, xx))

                # Also remove the cutoff segment itself
                dag_us.remove_node(xx)
            except KeyError:
                print('WARNING: nhm_segment {} does not exist in stream network'.format(xx))
    except TypeError:
        bandit_helper_log.error('\nSelected cutoffs should at least be an empty list instead of NoneType.')
        exit(200)

    bandit_helper_log.debug('Number of NHM upstream nodes (trimmed): {}'.format(dag_us.number_of_nodes()))
    bandit_helper_log.debug('Number of NHM upstream edges (trimmed): {}'.format(dag_us.number_of_edges()))

    # =======================================
    # Given a d/s segment (dsmost_seg) create a subset of u/s segments

    # Get all unique segments u/s of the starting segment
    uniq_seg_us = set()
    if dsmost_seg:
        for xx in dsmost_seg:
            try:
                pred = nx.dfs_predecessors(dag_us, xx)
                uniq_seg_us = uniq_seg_us.union(set(pred.keys()).union(set(pred.values())))
            except KeyError:
                bandit_helper_log.error('KeyError: Segment {} does not exist in stream network'.format(xx))
                # print('\nKeyError: Segment {} does not exist in stream network'.format(xx))

        # Get a subgraph in the dag_ds graph and return the edges
        dag_ds_subset = dag_ds.subgraph(uniq_seg_us).copy()

        # 2018-02-13 PAN: It is possible to have outlets specified which are not truly
        #                 outlets in the most conservative sense (e.g. a point where
        #                 the stream network exits the study area). This occurs when
        #                 doing headwater extractions where all segments for a headwater
        #                 are specified in the configuration file. Instead of creating
        #                 output edges for all specified 'outlets' the set difference
        #                 between the specified outlets and nodes in the graph subset
        #                 which have no edges is performed first to reduce the number of
        #                 outlets to the 'true' outlets of the system.
        node_outlets = [ee[0] for ee in dag_ds_subset.edges()]
        true_outlets = set(dsmost_seg).difference(set(node_outlets))
        bandit_helper_log.debug('node_outlets: {}'.format(','.join(map(str, node_outlets))))
        bandit_helper_log.debug('true_outlets: {}'.format(','.join(map(str, true_outlets))))

        # Add the downstream segments that exit the subgraph
        for xx in true_outlets:
            nhm_outlet = list(dag_ds.neighbors(xx))[0]
            dag_ds_subset.add_node(nhm_outlet, style='filled', fontcolor='white', fillcolor='grey')
            dag_ds_subset.add_edge(xx, nhm_outlet)
            dag_ds_subset.nodes[xx]['style'] = 'filled'
            dag_ds_subset.nodes[xx]['fontcolor'] = 'white'
            dag_ds_subset.nodes[xx]['fillcolor'] = 'blue'
    else:
        # No outlets specified so pull the CONUS
        dag_ds_subset = dag_ds

    return dag_ds_subset


def build_extraction(dag_ds_subset, nhm_params, hru_noroute, keep_hru_order):
    # dag_ds_subset = subset_stream_network(dag_ds, uscutoff_seg, dsmost_seg)

    # Create list of toseg ids for the model subset
    toseg_idx = list(set(xx[0] for xx in dag_ds_subset.edges))
    bandit_helper_log.info(f'Number of segments in subset: {len(toseg_idx)}')

    # Use the mapping to create subsets of nhm_seg, tosegment_nhm, and tosegment
    # NOTE: toseg_idx and new_nhm_seg are the same thing
    new_nhm_seg = [ee[0] for ee in dag_ds_subset.edges]

    # Using a dictionary mapping nhm_seg to 1-based index for speed
    new_nhm_seg_to_idx1 = OrderedDict((ss, ii+1) for ii, ss in enumerate(new_nhm_seg))

    # Generate the renumbered local tosegments (1-based with zero being an outlet)
    new_tosegment = [new_nhm_seg_to_idx1[ee[1]] if ee[1] in new_nhm_seg_to_idx1
                     else 0 for ee in dag_ds_subset.edges]

    # NOTE: With monolithic nhmParamDb files hru_segment becomes hru_segment_nhm and the
    # regional hru_segments are gone.
    # 2019-09-16 PAN: This initially assumed hru_segment in the monolithic paramdb was ALWAYS
    #                 ordered 1..nhru. This is not always the case so the nhm_id parameter
    #                 needs to be loaded and used to map the nhm HRU ids to their
    #                 respective indices.
    hru_segment = nhm_params.get('hru_segment_nhm').tolist()
    nhm_id = nhm_params.get('nhm_id').tolist()
    nhm_id_to_idx = nhm_params.get('nhm_id').index_map
    bandit_helper_log.info(f'Number of NHM hru_segment entries: {len(hru_segment)}')

    # Create a dictionary mapping hru_segment segments to hru_segment 1-based indices filtered by
    # new_nhm_seg and hru_noroute.
    seg_to_hru = OrderedDict()
    hru_to_seg = OrderedDict()

    for ii, vv in enumerate(hru_segment):
        # Contains both new_nhm_seg values and non-routed HRU values
        # keys are 1-based, values in arrays are 1-based
        if vv in new_nhm_seg:
            hid = nhm_id[ii]
            seg_to_hru.setdefault(vv, []).append(hid)
            hru_to_seg[hid] = vv
        elif nhm_id[ii] in hru_noroute:
            if vv != 0:
                err_txt = f'User-supplied non-routed HRU {nhm_id[ii]} routes to stream segment {vv} - Skipping.'
                bandit_helper_log.error(err_txt)
            else:
                hid = nhm_id[ii]
                seg_to_hru.setdefault(vv, []).append(hid)
                hru_to_seg[hid] = vv
    # print('{0} seg_to_hru {0}'.format('-'*15))
    # print(seg_to_hru)
    # print('{0} hru_to_seg {0}'.format('-'*15))
    # print(hru_to_seg)

    # HRU-related parameters can either be output with the legacy, segment-oriented order
    # or can be output maintaining their original relative order from the parameter database.
    if keep_hru_order:
        hru_order_subset = [kk for kk in hru_to_seg.keys()]

        new_hru_segment = [new_nhm_seg_to_idx1[kk] if kk in new_nhm_seg else 0 if kk == 0 else -1 for kk in
                           hru_to_seg.values()]
    else:
        # Get NHM HRU ids ordered by the segments in the model subset - entries are 1-based
        hru_order_subset = []
        for xx in new_nhm_seg:
            if xx in seg_to_hru:
                for yy in seg_to_hru[xx]:
                    hru_order_subset.append(yy)
            else:
                bandit_helper_log.warning(f'Stream segment {xx} has no HRUs connected to it.')

        # Append the additional non-routed HRUs to the list
        if len(hru_noroute) > 0:
            for xx in hru_noroute:
                if hru_segment[nhm_id_to_idx[xx]] == 0:
                    bandit_helper_log.info(f'User-supplied HRU {xx} is not connected to any stream segment')
                    hru_order_subset.append(xx)
                else:
                    err_txt = f'User-supplied HRU {xx} routes to stream segment ' + \
                              f'{hru_segment[nhm_id_to_idx[xx]]} - Skipping.'
                    bandit_helper_log.error(err_txt)

        # Renumber the hru_segments for the subset
        new_hru_segment = []

        for xx in new_nhm_seg:
            if xx in seg_to_hru:
                for _ in seg_to_hru[xx]:
                    # The new indices should be 1-based from PRMS
                    new_hru_segment.append(new_nhm_seg_to_idx1[xx])

        # Append zeroes to new_hru_segment for each additional non-routed HRU
        if len(hru_noroute) > 0:
            for xx in hru_noroute:
                if hru_segment[nhm_id_to_idx[xx]] == 0:
                    new_hru_segment.append(0)

    hru_order_subset0 = [nhm_id_to_idx[xx] for xx in hru_order_subset]
