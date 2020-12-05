import argparse  # 引入模块

if __name__ == '__main__':
    # 建立解析对象
    parser = argparse.ArgumentParser()

    parser.add_argument("echo")  # xx.add_argument("aa")
    # 给xx实例增加一个aa属性

    # 把parser中设置的所有"add_argument"给返回到args子类实例当中
    # 那么parser中增加的属性内容都会在args实例中，使用即可。
    args = parser.parse_args()
    parser.parse_args()
    print(arg.echo)  # 打印定位参数echo