# This forces Python to read these files and trigger their @register decorators
# the moment the 'equations' module is imported anywhere.

from . import sinebm
from . import flocking
from . import contxiong_lob
from . import contxiong_lob_jump
from . import contxiong_lob_mv
from . import contxiong_lob_adverse
