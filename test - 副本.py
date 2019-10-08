def test(f):
    print("before ...")
    f()
    print("after ...")



@test
def func():
    print("func was called")
