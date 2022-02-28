from options import Options
from paired import DraftDrawer

if __name__ == '__main__':
    opt = Options().get_options(eval=True)
    model = DraftDrawer(opt)
    model.test(opt.load_epoch)
