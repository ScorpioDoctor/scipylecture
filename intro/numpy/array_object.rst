..
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> import matplotlib.pyplot as plt
    >>> plt.switch_backend("Agg")


.. currentmodule:: numpy

NumPy 数组对象
======================

.. contents:: 本节内容
    :local:
    :depth: 1

什么是 NumPy 和 NumPy arrays?
--------------------------------

NumPy数组(arrays)
..................

:**Python** objects:

    - 高级数字对象: 整型数(integers), 浮点数(floats), ....

    - 容器: lists (插入和追加元素是无代价的), dictionaries(快速查找)

:**NumPy** provides:

    - 给Python提供了一个多维数组的扩展包

    - 更加的接近硬件(cpu,内存等) (高效)

    - 为科学计算而设计 (方便)

    - 被称为 面向数组的计算(*array oriented computing*),与Matlab类似

|

.. sourcecode:: pycon

    >>> import numpy as np
    >>> a = np.array([0, 1, 2, 3])
    >>> a
    array([0, 1, 2, 3])

.. tip::

    比如, 一个数组(array)包含:

    * 在离散时间步获得的实验/仿真数据

    * 由测量设备记录的信号, e.g. 声波

    * 一张图像的像素, 灰度或彩色

    * 在不同的 X-Y-Z 位置上测量到的位置数据, e.g. 核磁共振扫描数据

    * ...

**为啥它这么有用:** 内存高效的容器，提供了快速的数值运算。

.. sourcecode:: ipython

    In [1]: L = range(1000)

    In [2]: %timeit [i**2 for i in L]
    1000 loops, best of 3: 403 us per loop

    In [3]: a = np.arange(1000)

    In [4]: %timeit a**2
    100000 loops, best of 3: 12.7 us per loop


.. extension package to Python to support multidimensional arrays

.. diagram, import conventions

.. scope of this tutorial: drill in features of array manipulation in
   Python, and try to give some indication on how to get things done
   in good style

.. a fixed number of elements (cf. certain exceptions)
.. each element of same size and type
.. efficiency vs. Python lists

NumPy参考文档
..............................

- 在线文档网址: http://docs.scipy.org/

- 交互式帮助:

  .. sourcecode:: ipython

     In [5]: np.array?
     String Form:<built-in function array>
     Docstring:
     array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0, ...

  .. tip:

   .. sourcecode:: pycon

     >>> help(np.array) # doctest: +ELLIPSIS
     Help on built-in function array in module numpy.core.multiarray:
     <BLANKLINE>
     array(...)
         array(object, dtype=None, ...


- 用 `lookfor` 查找:

  .. sourcecode:: pycon

     >>> np.lookfor('create array') # doctest: +SKIP
     Search results for 'create array'
     ---------------------------------
     numpy.array
         Create an array.
     numpy.memmap
         Create a memory-map to an array stored in a *binary* file on disk.

  .. sourcecode:: ipython

     In [6]: np.con*?
     np.concatenate
     np.conj
     np.conjugate
     np.convolve

导入约定
..................

导入numpy的方式的约定:

.. sourcecode:: pycon

   >>> import numpy as np


创建数组
---------------

数组的手动创建
..............................

* **1-D**:

  .. sourcecode:: pycon

    >>> a = np.array([0, 1, 2, 3])
    >>> a
    array([0, 1, 2, 3])
    >>> a.ndim
    1
    >>> a.shape
    (4,)
    >>> len(a)
    4

* **2-D, 3-D, ...**:

  .. sourcecode:: pycon

    >>> b = np.array([[0, 1, 2], [3, 4, 5]])    # 2 x 3 array
    >>> b
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> b.ndim
    2
    >>> b.shape
    (2, 3)
    >>> len(b)     # 返回的是第一个纬度的size
    2

    >>> c = np.array([[[1], [2]], [[3], [4]]])
    >>> c
    array([[[1],
            [2]],
    <BLANKLINE>
           [[3],
            [4]]])
    >>> c.shape
    (2, 2, 1)

.. topic:: **练习: 简单数组**
    :class: green

    * 创建一个简单的二维数组。首先，重做上面的示例。然后创建你自己的：第一行的奇数倒计时，第二行的偶数如何？
    * 在这些数组上使用函数 :func:`len`, :func:`numpy.shape` 。他们返回的结果相互之间有什么内在联系呐？

可用于创建数组的函数
..............................

.. tip::

    在实践中，我们几乎不会一个一个的输入数组的每一项元素来创建数组 ...

* 创建均匀间隔的数组:

  .. sourcecode:: pycon

    >>> a = np.arange(10) # 0 .. n-1  (!)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> b = np.arange(1, 9, 2) # start, end (exclusive), step
    >>> b
    array([1, 3, 5, 7])

* 或 按点数创建数组:

  .. sourcecode:: pycon

    >>> c = np.linspace(0, 1, 6)   #参数： start, end, num-points
    >>> c
    array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
    >>> d = np.linspace(0, 1, 5, endpoint=False)  #不包括end位置上的数字
    >>> d
    array([ 0. ,  0.2,  0.4,  0.6,  0.8])

* 创建常见数组:

  .. sourcecode:: pycon

    >>> a = np.ones((3, 3))  # 必须记住: 传入的 (3, 3) 是一个元组(tuple)
    >>> a
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])
    >>> b = np.zeros((2, 2))
    >>> b
    array([[ 0.,  0.],
           [ 0.,  0.]])
    >>> c = np.eye(3)
    >>> c
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> d = np.diag(np.array([1, 2, 3, 4]))
    >>> d
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

* :mod:`np.random`: 创建随机数组 (Mersenne Twister PRNG):

  .. sourcecode:: pycon

    >>> a = np.random.rand(4)       # 在 [0, 1] 区间均匀分布
    >>> a  # doctest: +SKIP
    array([ 0.95799151,  0.14222247,  0.08777354,  0.51887998])

    >>> b = np.random.randn(4)      # 高斯分布(正态分布：normal distribution)
    >>> b  # doctest: +SKIP
    array([ 0.37544699, -0.11425369, -0.47616538,  1.79664113])

    >>> np.random.seed(1234)        # 设置随机数种子

.. topic:: **练习: 使用函数创建数组**
   :class: green

   * 练习它们吧！： ``arange``, ``linspace``, ``ones``, ``zeros``, ``eye`` 和 ``diag`` 。
   * 用随机数创建不同的数组.
   * 在创建随机数组之前尝试设置随机数种子.
   * 看看函数 ``np.empty`` 。 它干了什么？什么时候会派上用场？

.. EXE: construct 1 2 3 4 5
.. EXE: construct -5, -4, -3, -2, -1
.. EXE: construct 2 4 6 8
.. EXE: look what is in an empty() array
.. EXE: construct 15 equispaced numbers in range [0, 10]

基本数据类型
----------------

你可能注意到啦, 在一些例子中, 输出的数组元素后面会尾随着一个小点 (e.g. ``2.`` vs ``2``)。 
这是由于数据类型不一样导致的:

.. sourcecode:: pycon

    >>> a = np.array([1, 2, 3])
    >>> a.dtype
    dtype('int64')

    >>> b = np.array([1., 2., 3.])
    >>> b.dtype
    dtype('float64')

.. tip::

    不同的数据类型允许我们在内存中存储更紧凑的数据，但大多数时候我们只是使用浮点数。
    注意，在上面的例子中，NumPy自动从输入中检测数据类型。

-----------------------------

你可以显式的指定你想要的数据类型(data-type):

.. sourcecode:: pycon

    >>> c = np.array([1, 2, 3], dtype=float)
    >>> c.dtype
    dtype('float64')


**默认** 数据类型是浮点数(floating point):

.. sourcecode:: pycon

    >>> a = np.ones((3, 3))
    >>> a.dtype
    dtype('float64')

除了整型和浮点型，还有其他类型的:

:Complex:

  .. sourcecode:: pycon

        >>> d = np.array([1+2j, 3+4j, 5+6*1j])
        >>> d.dtype
        dtype('complex128')

:Bool:

  .. sourcecode:: pycon

        >>> e = np.array([True, False, False, True])
        >>> e.dtype
        dtype('bool')

:Strings:

  .. sourcecode:: pycon

        >>> f = np.array(['Bonjour', 'Hello', 'Hallo'])
        >>> f.dtype     # <--- strings containing max. 7 letters  # doctest: +SKIP
        dtype('S7')

:Much more:

    * ``int32``
    * ``int64``
    * ``uint32``
    * ``uint64``

.. XXX: mention: astype


基本可视化
-------------------

现在我们学会了创建arrays,接下来我们要可视化它们。

从启动 IPython 开始:

.. sourcecode:: bash

    $ ipython

或者用 notebook:

.. sourcecode:: bash

   $ ipython notebook

一旦 IPython 启动起来, 将会启用交互式绘图:

.. sourcecode:: pycon

    >>> %matplotlib  # doctest: +SKIP

或者, 从 notebook 启动, 则同样可以把图直接划在 notebook 上:

.. sourcecode:: pycon

    >>> %matplotlib inline # doctest: +SKIP

命令 ``inline`` 对 notebook 非常重要, 这个命令会把图显示在 notebook 上 而不是在一个新弹出的窗口上。

*Matplotlib* 是一个绘制 2D 图形的package。 我们可以像下面这样导入它的函数:

.. sourcecode:: pycon

    >>> import matplotlib.pyplot as plt  # 依紧凑的方式导入

然后使用 (请注意如果你没有通过 ``%matplotlib`` 命令启用交互式画图 ，你必须显式的的使用 ``show`` ):

.. sourcecode:: pycon

    >>> plt.plot(x, y)       # line plot    # doctest: +SKIP
    >>> plt.show()           # <-- 显示图 (如果使用交互式画图，则不需要这个) # doctest: +SKIP

或者, 如果你通过 ``%matplotlib`` 启用了交互式画图 :

.. sourcecode:: pycon

    >>> plt.plot(x, y)       # line plot    # doctest: +SKIP

* **1D plotting**:

.. sourcecode:: pycon

  >>> x = np.linspace(0, 3, 20)
  >>> y = np.linspace(0, 9, 20)
  >>> plt.plot(x, y)       # line plot    # doctest: +SKIP
  [<matplotlib.lines.Line2D object at ...>]
  >>> plt.plot(x, y, 'o')  # dot plot    # doctest: +SKIP
  [<matplotlib.lines.Line2D object at ...>]

.. image:: auto_examples/images/sphx_glr_plot_basic1dplot_001.png
    :width: 40%
    :target: auto_examples/plot_basic1dplot.html
    :align: center

* **2D arrays** (比如 图像):

.. sourcecode:: pycon

  >>> image = np.random.rand(30, 30)
  >>> plt.imshow(image, cmap=plt.cm.hot)    # doctest: +ELLIPSIS
  <matplotlib.image.AxesImage object at ...>
  >>> plt.colorbar()    # doctest: +ELLIPSIS
  <matplotlib.colorbar.Colorbar object at ...>

.. image:: auto_examples/images/sphx_glr_plot_basic2dplot_001.png
    :width: 50%
    :target: auto_examples/plot_basic2dplot.html
    :align: center

.. seealso:: 更多例子: :ref:`matplotlib chapter <matplotlib>`

.. topic:: **练习: 简单可视化**
   :class: green

   * 绘制一些简单的数组: 一个与时间相关的余弦函数，以及 一个 2D 矩阵。
   * 尝试在2D矩阵上使用 ``gray`` colormap 

.. * **3D plotting**:
..
..   For 3D visualization, we can use another package: **Mayavi**. A quick example:
..   start by **relaunching iPython** with these options: **ipython --pylab=wx**
..   (or **ipython -pylab -wthread** in IPython < 0.10).
..
..   .. image:: surf.png
..      :align: right
..      :scale: 60
..
..   .. sourcecode:: ipython
..
..       In [58]: from mayavi import mlab
..       In [61]: mlab.surf(image)
..       Out[61]: <enthought.mayavi.modules.surface.Surface object at ...>
..       In [62]: mlab.axes()
..       Out[62]: <enthought.mayavi.modules.axes.Axes object at ...>
..
..   .. tip::
..
..    The mayavi/mlab window that opens is interactive: by clicking on the
..    left mouse button you can rotate the image, zoom with the mouse wheel,
..    etc.
..
..    For more information on Mayavi :
..    https://github.enthought.com/mayavi/mayavi
..
..   .. seealso:: More in the :ref:`Mayavi chapter <mayavi-label>`


索引 与 切片
--------------------

Numpy数组的元素可以访问和赋值，就像Python的序列容器一样(e.g. lists):

.. sourcecode:: pycon

    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> a[0], a[2], a[-1]
    (0, 2, 9)

.. warning::

   索引从 0 开始, 就像 Python 的其他序列一样 (and C/C++).
   与此相反, 在 Fortran 或 Matlab, 索引从 1 开始。

python 的序列反转操作在Numpy中也是被支持的:

.. sourcecode:: pycon

   >>> a[::-1]
   array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

对于多维数组, 索引是整数构成的元组:

.. sourcecode:: pycon

    >>> a = np.diag(np.arange(3))
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 2]])
    >>> a[1, 1]
    1
    >>> a[2, 1] = 10 # third line, second column
    >>> a
    array([[ 0,  0,  0],
           [ 0,  1,  0],
           [ 0, 10,  2]])
    >>> a[1]
    array([0, 1, 0])


.. note::

  * 在 2D 数组中, 第一个维度对应于 **rows**, 第二个维度对应于 **columns**.
  * 对于多维数组 ``a``, ``a[0]`` 被解释为 把剩下的所有未指定的纬度上的元素全取出来.

**Slicing**: Arrays,像其他的Python序列一样可以被切片:

.. sourcecode:: pycon

    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> a[2:9:3] # [start:end:step]
    array([2, 5, 8])

注意最后一个索引是不包括的! :

.. sourcecode:: pycon

    >>> a[:4]
    array([0, 1, 2, 3])

切片中的三个参数不是都必须要同时具备的: 默认的, `start` 是 0, `end` 是最后一个，且 `step` is 1:

.. sourcecode:: pycon

    >>> a[1:3]
    array([1, 2])
    >>> a[::2]
    array([0, 2, 4, 6, 8])
    >>> a[3:]
    array([3, 4, 5, 6, 7, 8, 9])

下面是Numpy数组的索引和切片的一个小小的总结...

.. only:: latex

    .. image:: ../../pyximages/numpy_indexing.pdf
        :align: center

.. only:: html

    .. image:: ../../pyximages/numpy_indexing.png
        :align: center
        :width: 70%

你还可以将赋值与切片结合起来:

.. sourcecode:: pycon

   >>> a = np.arange(10)
   >>> a[5:] = 10
   >>> a
   array([ 0,  1,  2,  3,  4, 10, 10, 10, 10, 10])
   >>> b = np.arange(5)
   >>> a[5:] = b[::-1]
   >>> a
   array([0, 1, 2, 3, 4, 4, 3, 2, 1, 0])

.. topic:: **练习: 索引与切片**
   :class: green

   * 尝试不同形式的切片操作, 使用 ``start``, ``end`` 和 ``step``: 从 linspace 开始, try to obtain odd numbers
     counting backwards, and even numbers counting forwards.
   * 重新产生上表中的切片. 你可以使用一下表达式创建一个数组:

     .. sourcecode:: pycon

        >>> np.arange(6) + np.arange(0, 51, 10)[:, np.newaxis]
        array([[ 0,  1,  2,  3,  4,  5],
               [10, 11, 12, 13, 14, 15],
               [20, 21, 22, 23, 24, 25],
               [30, 31, 32, 33, 34, 35],
               [40, 41, 42, 43, 44, 45],
               [50, 51, 52, 53, 54, 55]])

.. topic:: **练习: Array 创建**
    :class: green

    创建下面的数组 (请注意数据类型要正确)::

        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 2],
         [1, 6, 1, 1]]

        [[0., 0., 0., 0., 0.],
         [2., 0., 0., 0., 0.],
         [0., 3., 0., 0., 0.],
         [0., 0., 4., 0., 0.],
         [0., 0., 0., 5., 0.],
         [0., 0., 0., 0., 6.]]

    Par on course: 3 statements for each

    *提示*: 单个的numpy数组元素可以被访问，类似于python的list, e.g. ``a[1]`` or ``a[1, 2]``。

    *提示*: 请查看 ``diag`` 函数的文档字符串。

.. topic:: 练习: 用于数组创建的平铺(Tiling)
    :class: green

    查看文档中的这个函数 ``np.tile``, 用它创建下面的这个数组::

        [[4, 3, 4, 3, 4, 3],
         [2, 1, 2, 1, 2, 1],
         [4, 3, 4, 3, 4, 3],
         [2, 1, 2, 1, 2, 1]]

拷贝 和 视图
----------------

一个切片操作在原数组上创建一个 **视图(view)** , 它只是访问数组数据的一种方法。因此，原数组并没有发生内存拷贝。
你可以使用 ``np.may_share_memory()`` 来检查两个数组是否共享了内存锁。然而，要注意这个方法的检查是启发式的，
可能会给你假正的结果(false positives)。

**当修改视图的时候, 原数组也会被修改**:

.. sourcecode:: pycon

    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> b = a[::2]
    >>> b
    array([0, 2, 4, 6, 8])
    >>> np.may_share_memory(a, b)
    True
    >>> b[0] = 12
    >>> b
    array([12,  2,  4,  6,  8])
    >>> a   # (!)
    array([12,  1,  2,  3,  4,  5,  6,  7,  8,  9])

    >>> a = np.arange(10)
    >>> c = a[::2].copy()  # force a copy
    >>> c[0] = 12
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    >>> np.may_share_memory(a, c)
    False



这种行为乍一看可能是令人惊讶的，…。但它可以节省内存和时间。


.. EXE: [1, 2, 3, 4, 5] -> [1, 2, 3]
.. EXE: [1, 2, 3, 4, 5] -> [4, 5]
.. EXE: [1, 2, 3, 4, 5] -> [1, 3, 5]
.. EXE: [1, 2, 3, 4, 5] -> [2, 4]
.. EXE: create an array [1, 1, 1, 1, 0, 0, 0]
.. EXE: create an array [0, 0, 0, 0, 1, 1, 1]
.. EXE: create an array [0, 1, 0, 1, 0, 1, 0]
.. EXE: create an array [1, 0, 1, 0, 1, 0, 1]
.. EXE: create an array [1, 0, 2, 0, 3, 0, 4]
.. CHA: archimedean sieve

.. topic:: 案例: 素数筛选
   :class: green

   .. image:: images/prime-sieve.png

   计算 0--99 内的素数, 使用筛子(sieve)

   * 创建一个 shape 为 (100,) 的 boolean 数组 ``is_prime``, 刚开始的时候全部填充为 True:

   .. sourcecode:: pycon

        >>> is_prime = np.ones((100,), dtype=bool)

   * 划出不属于素数的0和1:

   .. sourcecode:: pycon

       >>> is_prime[:2] = 0

   * 对每一个从 2 开始的整数 ``j`` , cross out its higher multiples:

   .. sourcecode:: pycon

       >>> N_max = int(np.sqrt(len(is_prime) - 1))
       >>> for j in range(2, N_max + 1):
       ...     is_prime[2*j::j] = False

   * 查看帮助文档 ``help(np.nonzero)``, 然后输出素数

   * Follow-up:

     - 将上面的代码移动到 ``prime_sieve.py``

     - 运行一下看是否正确

     - 使用下面推荐的方式优化 `the sieve of Eratosthenes
       <https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes>`_:

      1. Skip ``j`` which are already known to not be primes

      2. The first number to cross out is :math:`j^2`

花式索引(fancy indexing)
----------------------------

.. tip::

    NumPy数组可以用切片索引，也可以用布尔数组或整数数组(**掩码**)索引。
    这种方法称为花式索引(*fancy indexing*)。
    它创建副本而不是视图。

使用boolen型索引
...................

.. sourcecode:: pycon

    >>> np.random.seed(3)
    >>> a = np.random.randint(0, 21, 15)
    >>> a
    array([10,  3,  8,  0, 19, 10, 11,  9, 10,  6,  0, 20, 12,  7, 14])
    >>> (a % 3 == 0)
    array([False,  True, False,  True, False, False, False,  True, False,
            True,  True, False,  True, False, False], dtype=bool)
    >>> mask = (a % 3 == 0)
    >>> extract_from_a = a[mask] # or,  a[a%3==0]
    >>> extract_from_a           # 使用mask抽取一个子数组(sub-array)
    array([ 3,  0,  9,  6,  0, 12])

使用掩码索引对于为子数组分配新值非常有用:

.. sourcecode:: pycon

    >>> a[a % 3 == 0] = -1
    >>> a
    array([10, -1,  8, -1, 19, 10, 11, -1, 10, -1, -1, 20, -1,  7, 14])


使用整型数组进行索引
..................................

.. sourcecode:: pycon

    >>> a = np.arange(0, 100, 10)
    >>> a
    array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

索引可以用一个整数数组来完成，其中相同的索引可以重复几次:

.. sourcecode:: pycon

    >>> a[[2, 3, 2, 4, 2]]  # 注意: [2, 3, 2, 4, 2] 是一个 Python list
    array([20, 30, 20, 40, 20])

新的值可以用这种索引来赋值:

.. sourcecode:: pycon

    >>> a[[9, 7]] = -100
    >>> a
    array([   0,   10,   20,   30,   40,   50,   60, -100,   80, -100])

.. tip::

  当通过用整数数组索引新数组创建新数组时，新数组具有与整数数组相同的形状:

  .. sourcecode:: pycon

    >>> a = np.arange(10)
    >>> idx = np.array([[3, 4], [9, 7]])
    >>> idx.shape
    (2, 2)
    >>> a[idx]
    array([[3, 4],
           [9, 7]])


____

下图演示了各种高级索引应用

.. only:: latex

    .. image:: ../../pyximages/numpy_fancy_indexing.pdf
        :align: center

.. only:: html

    .. image:: ../../pyximages/numpy_fancy_indexing.png
        :align: center
        :width: 80%

.. topic:: **练习: 花式索引**
    :class: green

    * 同样，把上面的图表中所示的花哨索引重做一遍。
    * 使用左边的花式索引和右边的数组创建来将值赋值到数组中，例如，将上面图表中的数组的部分设置为零。


.. We can even use fancy indexing and :ref:`broadcasting <broadcasting>` at
.. the same time:
..
.. .. sourcecode:: pycon
..
..     >>> a = np.arange(12).reshape(3,4)
..     >>> a
..     array([[ 0,  1,  2,  3],
..            [ 4,  5,  6,  7],
..            [ 8,  9, 10, 11]])
..     >>> i = np.array([[0, 1], [1, 2]])
..     >>> a[i, 2] # same as a[i, 2*np.ones((2, 2), dtype=int)]
..     array([[ 2,  6],
..            [ 6, 10]])


