import torch

def d1_check():
    from task_d1 import prepare_test
    model = prepare_test()
    n = 2
    x = torch.rand(n, 3, 32, 32)
    y = model(x)
    assert y.shape == (n, 100)
    print('D1 check passed')

def d2_check():
    from task_d2 import prepare_test
    model = prepare_test()
    n = 2
    x = torch.rand(n, 3, 32, 32)
    y = model(x)
    assert y.shape == (n, 20)
    print('D2 check passed')

def d3_check():
    from task_d3 import prepare_test
    model = prepare_test()
    n = 2
    x = torch.rand(n, 3, 32, 32)
    yf, yc = model(x)
    assert yf.shape == (n, 100) and yc.shape == (n, 20)
    print('D3 check passed')

def d4_check(margins):
    from task_d4 import prepare_test
    n = 2
    x = torch.rand(n, 3, 32, 32)

    for m in margins:
        model_fine = prepare_test(m, True)
        model_coarse = prepare_test(m, False)

        for model in (model_fine, model_coarse):
            y = model(x)
            assert y.shape == (n, 576)

    print('D4 check passed')

def d5_test():
    from task_d5 import prepare_test
    model = prepare_test()
    n = 2
    x = torch.rand(n, 3, 32, 32)
    y = model(x)
    assert y.shape == (n, 100)
    print('D5 check passed')

def d7_check():
    from task_d7 import prepare_test
    model = prepare_test()
    n = 2
    x = torch.rand(n, 3, 32, 32)
    y = model(x)
    assert y.shape == (n, 576)
    print('D7 check passed')

if __name__ == '__main__':
    margins = (0.2, 0.5, 1.0)  # TODO adjust to a value for which you have stored the fine and coarse models for task D4
    d1_check()
    d2_check()
    d3_check()
    d4_check(margins)
    d5_test()
    d7_check()