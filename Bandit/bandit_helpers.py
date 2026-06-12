
import logging

from typing import Dict, List, Tuple

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



