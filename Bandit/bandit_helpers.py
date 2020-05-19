
import logging
import networkx as nx

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


def subset_stream_network(dag_ds, uscutoff_seg, dsmost_seg):
    # Create the upstream graph
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
