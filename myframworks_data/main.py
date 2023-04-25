import numpy as np
import csv

RND_MEAN = 0
RND_STD = 0.003
LEARNING_RATE = 0.0001


# csv -> numpy 배열로 변경
def load_dataset():
    global data, input_cnt, output_cnt
    # 파일은 어떻게 읽어오나요
    with open("./data/abalone.csv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)  # 한 줄 읽고 커서 다음 줄로 이동
        rows = []
        for row in csv_reader:
            rows.append(row)
    input_cnt, output_cnt = 10, 1
    data = np.zeros([len(rows), input_cnt + output_cnt])
    # print(data)

    # 인덱스 만들기 위해서 enumerate 사용
    # one hot encoding
    for n, row in enumerate(rows):
        if row[0] == "I":
            data[n, 0] = 1
        if row[0] == "M":
            data[n, 1] = 1
        if row[0] == "F":
            data[n, 2] = 1
        data[n, 3:] = row[1:]


# 모델은 기울기와 절편을 도출해내야 한다.
# 기준값이 필요하다
def init_model():
    global weight, bias, input_cnt, output_cnt
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])
    bias = np.zeros([output_cnt])
    pass


def train_and_test(epoch_count=10, mb_size=10):
    step_count = arrange_data(mb_size)
    text_x, test_y = get_data_test()

    # 1. 데이터를 나눈다. (8대2)
    # 2. 학습(행렬곱). 넘파이 ---- 손실과 정확도라는 지표 획득
    for epoch in range(epoch_count):
        losses = []
        accs = []

        for n in range(step_count):
            train_x, train_y = get_train_test(mb_size, n)
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)

    # 3. 테스트

def run_train(x,y):
    # 순방향 => 정확도
    output, aux_nn = forward_neuralnet(x)
    loss, aux_pp = forward_postproc(output, y)
    accuracy = eval_accuracy(output, y)

    # 역방향 => 오차
    G_loss = 1.0
    G_output = backprop_postproc(G_loss, aux_pp)
    backprops_neuralnet(G_output, aux_nn)

    return loss, accuracy



def get_train_data(mb_size, nth):
    global data, shuffle_map, test_begin_idx, output_cnt
    if nth == 0:
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size * nth: mb_size * (nth + 1)]]
    return train_data[:, :-output_cnt], train_data[:, -output_cnt:]


def arrange_data(mb_size):
    global data, shuffle_map, test_begin_idx
    shuffle_map = np.arange(data.shape[0])  # arange - 배열 만들어줌
    np.random.shuffle(shuffle_map)
    step_count = int(data.shape[0] * 0.8) // mb_size
    test_begin_idx = step_count * mb_size
    return step_count


def get_data_test():
    global data, shuffle_map, test_begin_idx, output_cnt
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]


if __name__ == "__main__":
    # 컴프 -> 입,처,출
    load_dataset() # CSV 파일 읽는 방법
    init_model() # 기울기와 절편
    # print(data)
    # print(weight)  # 10행 1열이 아닌 1행 10열임. numpy라서. 뒤에서부터 읽는다.
    # print(bias)
    train_and_test() # Epoch, Batch(지역검색 사이즈 결정. 경사하강법)
    print(data)
