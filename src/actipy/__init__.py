name = "actipy"
__author__ = "Shing Chan, Aiden Doherty"
__email__ = "shing.chan@ndph.ox.ac.uk, aiden.doherty@ndph.ox.ac.uk"
__maintainer__ = "Shing Chan"
__maintainer_email__ = "shing.chan@ndph.ox.ac.uk"
__license__ = "See LICENSE.md"

from actipy.reader import read_device, process

from . import _version
__version__ = _version.get_versions()['version']
