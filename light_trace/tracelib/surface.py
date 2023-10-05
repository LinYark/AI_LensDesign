

class surface_lib():
    def __init__(self, 
                r        = 'inf',
                t        = 10,
                h        = 50,
                n        = 'air',
                comment  = "none",
                ):
        self.r        = r
        self.t        = t
        self.h        = h
        self.n        = n
        self.comment  = comment

if __name__=="__main__":
    s = surface_lib()
    a = 1