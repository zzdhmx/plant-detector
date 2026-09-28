"""支持 python -m plantdet -i 图片。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
