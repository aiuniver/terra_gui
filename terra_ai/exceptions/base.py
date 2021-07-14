class TerraBaseException(Exception):
    class Meta:
        message = "Undefined error"

    def __init__(self, *args, **kwargs):
        if len(args):
            __message = args[0]
            args = args[1:]
        else:
            __message = self.Meta.message
        super().__init__(__message, *args, **kwargs)
