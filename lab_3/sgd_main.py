from ucimlrepo import fetch_ucirepo
from metrics import benchmark
from keras_funcs import run as keras_run
from torch_funcs import run as torch_run

from stochastic import sgd


def scale_data(x_df, y_df, x_scale=None, y_scale=None):
    if x_scale is None:
        x_scaled = (x_df - x_df.mean()) / x_df.std()
    else:
        x_scaled = x_df / x_scale

    if y_scale is None:
        y_scaled = (y_df - y_df.mean()) / y_df.std()
    else:
        y_scaled = y_df / y_scale
    return x_scaled, y_scaled


if __name__ == '__main__':
    poker_hand = fetch_ucirepo(id=158)

    x = poker_hand.data.features
    y = poker_hand.data.targets

    cols = []
    for i in range(1, 6):
        cols += [f"suit_{i}", f"rank_{i}"]
    x.columns = cols
    y.columns = ["hand"]

    x, y = scale_data(x, y, y_scale=9)

    print(x.head())
    print(y.head())

    # trainer = sgd()
    # w, b, hist = trainer.fit(x, y)
    # print(hist[-1])

    w, b, fb = benchmark(x.values[:10000], y.values[:10000])
    print(fb, sep='\n')
    w, b, fb = benchmark(x.values[:10000], y.values[:10000], cfg={"reg": 'l1'})
    print(fb, sep='\n')
    w, b, fb = benchmark(x.values[:10000], y.values[:10000], cfg={"reg": 'l2'})
    print(fb, sep='\n')
    w, b, fb = benchmark(x.values[:10000], y.values[:10000], cfg={"reg": 'elastic'})
    print(fb, sep='\n')

    # keras_run(x.values[:10000], y.values[:10000])
    # torch_run(x.values[:10000], y.values[:10000])
