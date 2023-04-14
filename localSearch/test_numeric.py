import numeric

def test_create_problem():
    result = numeric.create_problem("./data/Convex.txt")
    test_case1 = result[0]
    test_case2 = (len(result[1]), len(result[1][0]))
    test_case1_1 = "(x1 - 2) ** 2 +5 * (x2 - 5) ** 2 + 8 * (x3 + 8) ** 2 + 3 * (x4 + 1) ** 2 + 6 * (x5 - 7) ** 2"
    test_case2_1 = (3,5)     #[0:-1] #개행문자 제거하면 스트링 같게 만들 수 있다. 원래 함수에서 제거해도 되고
    print(test_case1)
    print(test_case2)
    assert test_case1 == test_case1_1
    assert test_case2 == test_case2_1

