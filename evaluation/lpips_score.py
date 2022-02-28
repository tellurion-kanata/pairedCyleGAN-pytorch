import lpips
import datasets

def calc_lpips_score(dataset, device):
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    dscore = 0.
    for idx, data in enumerate(dataset):
        predict, reference = data['A'].to(device), data['B'].to(device)
        dscore += loss_fn_alex(predict, reference).sum()

    dscore = dscore / len(dataset)
    return dscore


def lpips_score(dataroot, batch_size=50, device='cpu', num_threads=8, pattern='*.png'):
    dataloader = datasets.EvalDataLoader()
    dataloader.initialize(dataroot, batch_size, num_threads, pattern)
    dataset = dataloader.load_data()
    score = calc_lpips_score(dataset, device)
    print('LPIPS score %.7f: ' % score)