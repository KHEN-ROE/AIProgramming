import numpy as np
import csv

RND_MEAN = 0
RND_STD = 0.003
LEARNING_RATE = 0.0001


def load_dataset():
    global data, input_cnt, ouput_cnt
    with open('./data/abalone.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)
        rows = []
        for row in csv_reader:
            rows.append(row)
    input_cnt, ouput_cnt = 10, 1
    data = np.zeros([len(rows), input_cnt + ouput_cnt])

    for n, row in enumerate(rows):
        if row[0] == "I":
            data[n, 0] = 1
        if row[0] == "M":
            data[n, 1] = 1
        if row[0] == "F":
            data[n, 2] = 1
        data[n, 3:] = row[1:]


def init_model():
    global weight, bias, input_cnt, ouput_cnt
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, ouput_cnt])
    bias = np.zeros([ouput_cnt])


def train_and_test(epoch_count=10, mb_size=10):
    step_cout = arrange_data(mb_size)
    test_x, test_y = get_data_test()

    # 2. 학습
    for epoch in range(epoch_count):
        losses = []
        accs = []

        for n in range(step_cout):
            acc = run_test(test_x, test_y)
            train_x, train_y = get_train_data(mb_size, n)
            loss, acc = run_train(train_x, train_y)  # train_x에 행렬 들어옴.
            losses.append(loss)
            accs.append(acc)

        print(
            f"Epoch {epoch + 1} : loss={np.mean(losses)},"
            f" accs={np.mean(accs):5.3f}, acc={np.mean(acc):5.3f}")  # accs는 정확도인데, 무슨 정확도? epoch당 정확도

    # 3. 테스트
    final_acc = run_test(test_x, test_y)
    print(f"Acc={np.mean(final_acc):5.3f}") # Acc 평균 = 80 정도 나옴


def run_test(x, y):
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy


def run_train(x, y):
    # 순방향 => 정확도
    output, aux_nn = forward_neuralnet(x)  # output, x -> 정확도 계산
    loss, aux_pp = forward_postproc(output, y)  # output, x -> 사실 y가 아닌 x
    accuracy = eval_accuracy(output, y)

    # 역방향 => 오차
    G_loss = 1.0
    G_output = backprop_postproc(G_loss, aux_pp)
    backprop_neuralnet(G_output, aux_nn)  # 기울기와 절편을 여기서 수정

    return loss, accuracy


def forward_neuralnet(x):
    # w1*x1 + b
    global weight, bias
    output = np.matmul(x, weight) + bias  # matmul - 행렬 곱함.

    return output, x


def forward_postproc(output, y):
    diff = output - y  # output - x
    square = np.square(diff)
    loss = np.mean(square)
    return loss, diff


def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output - y) / y))
    return 1 - mdiff


# 오차의 기울기 구함
def backprop_postproc(G_loss, diff):  # diff => x
    shape = diff.shape

    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff
    g_diff_output = 1

    G_square = g_loss_square * G_loss
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff  # G_diff - 근사차이

    return G_output  # 편미문 값


def backprop_neuralnet(G_output, x):
    global weight, bias
    g_output_w = x.transpose()

    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis=0)

    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b


def get_train_data(mb_size, nth):
    global data, shuffle_map, test_begin_idx, ouput_cnt
    if nth == 0:
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size * nth: mb_size * (nth + 1)]]
    return train_data[:, :-ouput_cnt], train_data[:, -ouput_cnt:]


def arrange_data(mb_size):
    global data, shuffle_map, test_begin_idx
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)
    step_cout = int(data.shape[0] * 0.8) // mb_size
    test_begin_idx = step_cout * mb_size
    return step_cout


def get_data_test():
    global data, shuffle_map, test_begin_idx, ouput_cnt
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:, :-ouput_cnt], test_data[:, -ouput_cnt:]


if __name__ == "__main__":
    load_dataset()  # CSV 파일 읽는 방법
    init_model()  # 기울기와 절편
    train_and_test()  # Epoch, Batch
