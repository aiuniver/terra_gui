
class CascadeBlock:

    def get(self, type_, **kwargs):
        return self.__getattribute__(type_)(**kwargs)