# 程序规范

本项目由两个部分组成

- application
- algorithm

## Workflow

1. @Jiaming 负责 application 实现。DDL：5.15
2. 其他同学需要实现 1个或者多个路径算法 DDL：5.17
3. @Jiaming 需要把 算法改写为算法实现 DDL 5.19
4. 其他同学负责 Report 写作，项目的文档，以及 Presentation

## Algorithm Requirement

### 算法的目标

根据应用程序给出的参数，算法的 route() 需要 return 从一个地点到另一个地点的 路线。

### 参数与 return 值的规范

application 将会调用 

### 要求

- 需要使用 Python 或者 Java 实现
- **需要原生实现算法，不可以使用内置函数或第三方函数**
- 可以自定义其他函数

需要遵循下面的模板规范

```python

# 可以定义一些全局变量
# 可以定义其他函数

# @Jiaming 的 Application 只会调用 route() 这个函数
def route(stations_base, stations_packages_number,
          routes_packages_number, package_from, package_to,
          package_priority):
  
  
```

### 提交要求

每个算法放到单独的 py 文件中，放到 algorithm 文件夹。

## Application Requirement

由 @Jiaming 自行规范。







- A -> B 的最佳路径 ，应该走哪些站点 []，不管啥时候问

 Box

- 当前 每个站点 有多少包裹正在处理 int
- 每个路径上 有多少包裹正在运输 int
- 所请求的包裹的优先级
  - 快运
  - 满运

---

[ A ([x, y],  station/center) ]

## Reference

1. https://juejin.cn/s/python%E8%BF%90%E8%A1%8Cpy%E6%96%87%E4%BB%B6%E5%B8%A6%E5%8F%82%E6%95%B0