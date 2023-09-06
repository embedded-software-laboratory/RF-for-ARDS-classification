from typing import Any



from filtering.IFilter import IFilter




class ukaFilter(IFilter):

    def __init__(self, options: Any):
        super().__init__(options)