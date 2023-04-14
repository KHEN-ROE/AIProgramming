#First-Choice
## 문제 해결 방법

# 간단한 언덕 등반 알고리즘
## Convex.txt 를 사용해서 계산
## 1. 파일을 읽어서 뭔가 만들어냄
## 2. 그 뭔가로 초기값 생성
## 3. Convex.txt 파일의 수식과 값을 이용해서 계산

#파이썬은 함수명 전부 언더바

import random

def create_problem(filename):
    # 1-1. 파일을 읽자
    ini_file = open(filename, 'r')
    expression = ini_file.readline().strip() # strip() - 개행문자 제거
    # 1-2. 수식과 리스트로 분리
    var_names = []
    lows = []
    up = []

    for line in ini_file.readlines():
       # n,l,u = tuple(line.split())
        var_names.append(line.split(",")[0])
        lows.append(float(line.split(",")[1]))
        up.append(float(line.split(",")[2]))

    ini_file.close()
    domain = [var_names, lows, up]
    # 1-3. 리턴
    return (expression, domain)

def random_init(p):
    expression, domain = p
    init = []
    for i in range(0, len(domain[0])):
        # 최대 최소 사이의 랜덤 값
        init.append(random.uniform(domain[1][i], domain[2][i]))

    return init

def evaluate(state, p):
    num_eval = 0
    expression, _ = p[0] # p가 튜플이라고 간주
    var_name = p[1][0]

    for i in range(len(var_name)):
        assignment = var_name[i] + '=' + str(state[i])
        exec(assignment) # exec - 문자열을 실행시키는 함수. 자바의 리플렉션

    return eval(expression)

if __name__ == "__main__":
    print(create_problem("./data/Convex.txt"))
    #pass
    

