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

实现 Option 1 或者 Option 2，任何一个都可以。

要求

- 需要使用 Python 或者 Java 实现
- **需要原生实现算法，不可以使用内置函数或第三方函数**
- 可以自定义其他函数

> 说明：为了更加规范每个站点的类型，我们在教师的规范上补充了约定。
>
> 我们的物流系统中有两种类型的地点： `Stations` , `Centers`, `Airport`.
>
> - Stations 能收件投件，也可以用于中转
> - Centers 只能用于中转。
> - Airport 只能用于中转。
>
> 约定为
>
> - Stations 规模较小，中转能力较弱。
> - Centers 规模较大，中转能力较强。 
> - Airport  规模较大，中转能力较强
>
> ~~根据教师约定，任意两点之间都有路径。~~但为了方便设计程序，我们约定
>
> - 两个 Airport 之间的运输只能是飞机运输。
> - 起点和终点都只能是 station。
>
> 表示方法
>
> - s1, s2, s3： 表示 station
> - c1, c2, c3： 表示 center
> - a1, a2, a3: 表示 airport
>
> 其他参考因素（如有需要）：在模拟器程序中
>
> - 运输的最短时间是 1分钟
> - 陆运速度是 1个单位/每分钟
> - 空运速度是 10个单位/每分钟
>
> 其余可根据自己的规划，自行斟酌标准。

### Option 1: route()

#### 算法的目标

根据应用程序给出的参数，算法的 route() 需要 return 从一个地点到另一个地点的 路线。当我请求的时候你就理解为, 此包裹还未发出, 我只是想了解你的路线规划。

#### 参数与 return 值的规范

application 将会调用 route() 这个方法。

- 需要遵循下面的模板规范

```python

# 可以定义一些全局变量
# 可以定义其他函数

# @Jiaming 的 Application 只会调用 route() 这个函数
def route(stations_base, stations_packages_number,
          routes_packages_number, package_from, package_to,
          package_priority):
  # 分析路径选择
  routes = xxx;
  return routes
# 不是所有的参数都需要使用，比如 stations_packages_number 和 routes_packages_number 可能用不到。
```

其中

stations_base: 所有的站点，类型 dict (Python 字典，类似 C++ 的 Map。)案例如下

```python
# key 是 站点名称，string
# value 是站点位置，使用坐标表示，使用 元组(int, int) 表示
{
  "s1": (2, 6),
  "s2": (2, 3),
  "c1": (4, 5),
  "a1": (8, 6)
}
```

stations_packages_number: 每个站点的包裹数量。类型 dict (Python 字典，类似 C++ 的 Map。)案例如下

```python
# key 是 站点名称，string
# value 是站点位置，int
{
  "s1": 605,
  "s2": 21,
  "c1": 988,
  "a1": 225
}
```

routes_packages_number: 两个站点之间的有向路径上的正在运输的包裹数量。类型 dict (Python 字典，类似 C++ 的 Map。)案例如下

```python
# key 是 有向路径名称，eg. "s1->s2", string
# value 是 有向路径 上的正在运输的包裹的数量。
# 不考虑同城快递，如 "s1->s1"
{
  "s1->s2": 223,
  "a2->a1": 281,
  "a1->a2": 332
}
```

package_from： 起始站点，只能是 station,只能是 station, string, eg. "s1"

package_to: 终点站点，只能是 station, string, eg. "s2"

package_priority: 包裹优先级

- 0 = 标准件
- 1 = 速运件

返回值约定：数据类型 list

```
# 如果 s1 是起点, s9 是终点
# 返回的路径应当如下 list of string
["s1", "c1", "c3", "c5", "s9"]
```

## Option2: nexthop()

#### 算法的目标

根据应用程序给出的参数，算法的 nexthop () 需要 return 这个包裹的下一个节点应该是哪里。即根据当前包裹的位置以及其他潜在的因素，判断下一步包裹应该往哪里走。

#### 参数与 return 值的规范

application 将会调用 nexthop() 这个方法。

- 需要遵循下面的模板规范

```python
# 可以定义一些全局变量
# 可以定义其他函数

# @Jiaming 的 Application 只会调用 nexthop() 这个函数
def nexthop(stations_base, stations_packages_number,
          routes_packages_number, package_current_node,,
          package_priority):
  # 分析下一跳的选择
  next = xxx;
  return routes
# 不是所有的参数都需要使用，比如 stations_packages_number 和 routes_packages_number 可能用不到。
```

其中

stations_base: 所有的站点，类型 dict (Python 字典，类似 C++ 的 Map。)案例如下

```python
# key 是 站点名称，string
# value 是站点位置，使用坐标表示，使用 元组(int, int) 表示
{
  "s1": (2, 6),
  "s2": (2, 3),
  "c1": (4, 5),
  "a1": (8, 6)
}
```

stations_packages_number: 每个站点的包裹数量。类型 dict (Python 字典，类似 C++ 的 Map。)案例如下

```python
# key 是 站点名称，string
# value 是站点位置，int
{
  "s1": 605,
  "s2": 21,
  "c1": 988,
  "a1": 225
}
```

routes_packages_number: 两个站点之间的有向路径上的正在运输的包裹数量。类型 dict (Python 字典，类似 C++ 的 Map。)案例如下

```python
# key 是 有向路径名称，eg. "s1->s2", string
# value 是 有向路径 上的正在运输的包裹的数量。
# 不考虑同城快递，如 "s1->s1"
{
  "s1->s2": 223,
  "a2->a1": 281,
  "a1->a2": 332
}
```

package_current_node: package 当前的站点，, string, eg. "s1"



package_priority: 包裹优先级

- 0 = 标准件
- 1 = 速运件

返回值约定：数据类型 list

```
# 如果 s1 是起点, s9 是终点
# 返回的路径应当如下 list of string
["s1", "c1", "c3", "c5", "s9"]
```

## 提交要求

每个算法放到单独的 py 文件中，放到 algorithm 文件夹。

备注好是 Option 1 还是 Option 2.

## Application Requirement

由 @Jiaming 自行规范。

## Reference

1. https://juejin.cn/s/python%E8%BF%90%E8%A1%8Cpy%E6%96%87%E4%BB%B6%E5%B8%A6%E5%8F%82%E6%95%B0