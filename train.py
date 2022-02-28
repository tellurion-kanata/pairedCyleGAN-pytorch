from options import Options
from paired import DraftDrawer

if __name__ == '__main__':
    opt = Options().get_options()
    model = DraftDrawer(opt)
    model.train()
