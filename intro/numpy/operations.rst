
..  For doctests
    
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> # For doctest on headless environments
    >>> import matplotlib.pyplot as plt
    >>> plt.switch_backend("Agg")

.. currentmodule:: numpy

数组上的数值操作
==============================

.. contents:: Section contents
    :local:
    :depth: 1


按元素操作符
----------------------

基本操作符
................

与标量运算:

.. sourcecode:: pycon

    >>> a = np.array([1, 2, 3, 4])
    >>> a + 1
    array([2, 3, 4, 5])
    >>> 2**a
    array([ 2,  4,  8, 16])

所有算术运算都是按元素操作的:

.. sourcecode:: pycon

    >>> b = np.ones(4) + 1
    >>> a - b
    array([-1.,  0.,  1.,  2.])
    >>> a * b
    array([ 2.,  4.,  6.,  8.])

    >>> j = np.arange(5)
    >>> 2**(j + 1) - j
    array([ 2,  3,  6, 13, 28])

Numpy的这些算术运算当然比用纯Python的相关操作快的多啦:

.. sourcecode:: pycon

   >>> a = np.arange(10000)
   >>> %timeit a + 1  # doctest: +SKIP
   10000 loops, best of 3: 24.3 us per loop
   >>> l = range(10000)
   >>> %timeit [i+1 for i in l] # doctest: +SKIP
   1000 loops, best of 3: 861 us per loop


.. warning:: **数组乘法(Array multiplication)不是矩阵乘法(matrix multiplication):**

    .. sourcecode:: pycon

        >>> c = np.ones((3, 3))
        >>> c * c                   # 不是 矩阵乘法
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 1.,  1.,  1.]])

.. note:: **矩阵乘法:**

    .. sourcecode:: pycon

        >>> c.dot(c)
        array([[ 3.,  3.,  3.],
               [ 3.,  3.,  3.],
               [ 3.,  3.,  3.]])

.. topic:: **练习: 按元素操作**
   :class: green

    * 尝试简单的算术按元素操作：用奇数元素加偶数元素
    * 使用 ``%timeit`` 对比纯Python和Numpy的相关运算的效率.
    * 生成:

      * ``[2**0, 2**1, 2**2, 2**3, 2**4]``
      * ``a_j = 2^(3*j) - j``


其他操作符
................

**比较运算:**

.. sourcecode:: pycon

    >>> a = np.array([1, 2, 3, 4])
    >>> b = np.array([4, 2, 2, 4])
    >>> a == b
    array([False,  True, False,  True], dtype=bool)
    >>> a > b
    array([False, False,  True, False], dtype=bool)

.. tip::

   按数组比较(Array-wise comparisons):

   .. sourcecode:: pycon

    >>> a = np.array([1, 2, 3, 4])
    >>> b = np.array([4, 2, 2, 4])
    >>> c = np.array([1, 2, 3, 4])
    >>> np.array_equal(a, b)
    False
    >>> np.array_equal(a, c)
    True


**逻辑操作:**

.. sourcecode:: pycon

    >>> a = np.array([1, 1, 0, 0], dtype=bool)
    >>> b = np.array([1, 0, 1, 0], dtype=bool)
    >>> np.logical_or(a, b)
    array([ True,  True,  True, False], dtype=bool)
    >>> np.logical_and(a, b)
    array([ True, False, False, False], dtype=bool)

**超越函数:**

.. sourcecode:: pycon

    >>> a = np.arange(5)
    >>> np.sin(a)
    array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ])
    >>> np.log(a)
    array([       -inf,  0.        ,  0.69314718,  1.09861229,  1.38629436])
    >>> np.exp(a)
    array([  1.        ,   2.71828183,   7.3890561 ,  20.08553692,  54.59815003])


**形状不匹配**

.. sourcecode:: pycon

    >>> a = np.arange(4)
    >>> a + np.array([1, 2])  # doctest: +SKIP
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: operands could not be broadcast together with shapes (4) (2)

*广播?* 我们稍后会讲到的啦 :ref:`later <broadcasting>`.

**转置:**

.. sourcecode:: pycon

    >>> a = np.triu(np.ones((3, 3)), 1)   # 请看 help(np.triu)
    >>> a
    array([[ 0.,  1.,  1.],
           [ 0.,  0.,  1.],
           [ 0.,  0.,  0.]])
    >>> a.T
    array([[ 0.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  1.,  0.]])


.. warning:: **转置是一个视图(view)**

    因此, 下面的代码 **is wrong** 而且 **不会创建一个对称矩阵的**::

        >>> a += a.T

    上述代码在比较小的数组上是可以工作的 (因为有缓存)， 但是对大数组会失败，结果不可预测。

.. note:: **线性代数**

    子模块 :mod:`numpy.linalg` 实现了基本线性代数, 比如求解线性系统，奇异值分解(SVD)，等等。
    然而该子模块并不保证它们的实现程序是高效的，因此我梦推荐使用 :mod:`scipy.linalg`, 
    这个模块在章节 :ref:`scipy_linalg` 中有介绍。

.. topic:: 练习一些其他的操作
   :class: green

    * 查看 ``np.allclose`` 的帮助文档. 这个函数啥时候有用武之地呢？
    * 接着查看这两个函数 ``np.triu`` 和 ``np.tril`` 。


基本约简
----------------

计算和
..............

.. sourcecode:: pycon

    >>> x = np.array([1, 2, 3, 4])
    >>> np.sum(x)
    10
    >>> x.sum()
    10

.. image:: images/reductions.png
    :align: right
    :target: http://www.studyai.com

按行求和与按列求和:

.. sourcecode:: pycon

    >>> x = np.array([[1, 1], [2, 2]])
    >>> x
    array([[1, 1],
           [2, 2]])
    >>> x.sum(axis=0)   # 列 (第一纬)
    array([3, 3])
    >>> x[:, 0].sum(), x[:, 1].sum()
    (3, 3)
    >>> x.sum(axis=1)   # 行 (第二维)
    array([2, 4])
    >>> x[0, :].sum(), x[1, :].sum()
    (2, 4)

.. tip::

  在更高维空间中的操作:

  .. sourcecode:: pycon

    >>> x = np.random.rand(2, 2, 2)
    >>> x.sum(axis=2)[0, 1]     # doctest: +ELLIPSIS
    1.14764...
    >>> x[0, 1, :].sum()     # doctest: +ELLIPSIS
    1.14764...

其他约简操作
................

--- 工作方式是一样的 (都接受参数 ``axis=``)

**Extrema:**

.. sourcecode:: pycon

  >>> x = np.array([1, 3, 2])
  >>> x.min()
  1
  >>> x.max()
  3

  >>> x.argmin()  # 最小值的索引
  0
  >>> x.argmax()  # 最大值的索引
  1

**逻辑操作:**

.. sourcecode:: pycon

  >>> np.all([True, True, False])
  False
  >>> np.any([True, True, False])
  True

.. note::

   可以用于数组比较:

   .. sourcecode:: pycon

      >>> a = np.zeros((100, 100))
      >>> np.any(a != 0)
      False
      >>> np.all(a == a)
      True

      >>> a = np.array([1, 2, 3, 2])
      >>> b = np.array([2, 2, 3, 2])
      >>> c = np.array([6, 4, 4, 5])
      >>> ((a <= b) & (b <= c)).all()
      True

**统计:**

.. sourcecode:: pycon

  >>> x = np.array([1, 2, 3, 1])
  >>> y = np.array([[1, 2, 3], [5, 6, 1]])
  >>> x.mean()
  1.75
  >>> np.median(x)
  1.5
  >>> np.median(y, axis=-1) # 最后一个数轴
  array([ 2.,  5.])

  >>> x.std()          # 全部总体标准差.
  0.82915619758884995


... and many more (best to learn as you go).

.. topic:: **练习: 约减**
   :class: green

    * 假定有这样一个函数 ``sum``, 你还希望有哪些函数呢?
    * ``sum`` 和 ``cumsum`` 之间的区别是啥?

.. topic:: 工作案例: 数据统计
   :class: green

   在这儿下载数据 :download:`populations.txt <../../data/populations.txt>`
   描述了20年来加拿大北部野兔和山猫(和胡萝卜)的数量。

   你可以在编辑器中查看下载的数据, 或者 在 IPython 中查看(在windows上不能用):

   .. sourcecode:: ipython

     In [1]: !cat data/populations.txt

   首先，把数据加载到 NumPy 数组:

   .. sourcecode:: pycon

     >>> data = np.loadtxt('data/populations.txt')
     >>> year, hares, lynxes, carrots = data.T  # 技巧: columns to variables

   然后画出它们:

   .. sourcecode:: pycon

     >>> from matplotlib import pyplot as plt
     >>> plt.axes([0.2, 0.1, 0.5, 0.8]) # doctest: +SKIP
     >>> plt.plot(year, hares, year, lynxes, year, carrots) # doctest: +SKIP
     >>> plt.legend(('Hare', 'Lynx', 'Carrot'), loc=(1.05, 0.5)) # doctest: +SKIP

   .. image:: auto_examples/images/sphx_glr_plot_populations_001.png
      :width: 50%
      :target: auto_examples/plot_populations.html
      :align: center

   随时间变化的种群平均数量:

   .. sourcecode:: pycon

     >>> populations = data[:, 1:]
     >>> populations.mean(axis=0)
     array([ 34080.95238095,  20166.66666667,  42400.        ])

   随时间变化的样本标准差:

   .. sourcecode:: pycon

     >>> populations.std(axis=0)
     array([ 20897.90645809,  16254.59153691,   3322.50622558])

   每年种群的数量最大的是哪一个?:

   .. sourcecode:: pycon

     >>> np.argmax(populations, axis=1)
     array([2, 2, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 2, 2, 2, 2, 2])

.. topic:: 工作案例: diffusion using a random walk algorithm

  .. image:: random_walk.png
     :align: center

  .. tip::

    Let us consider a simple 1D random walk process: at each time step a
    walker jumps right or left with equal probability.

    We are interested in finding the typical distance from the origin of a
    random walker after ``t`` left or right jumps? We are going to
    simulate many "walkers" to find this law, and we are going to do so
    using array computing tricks: we are going to create a 2D array with
    the "stories" (each walker has a story) in one direction, and the
    time in the other:

  .. only:: latex

    .. image:: random_walk_schema.png
        :align: center

  .. only:: html

    .. image:: random_walk_schema.png
        :align: center
        :width: 100%

  .. sourcecode:: pycon

   >>> n_stories = 1000 # number of walkers
   >>> t_max = 200      # time during which we follow the walker

  We randomly choose all the steps 1 or -1 of the walk:

  .. sourcecode:: pycon

   >>> t = np.arange(t_max)
   >>> steps = 2 * np.random.randint(0, 1 + 1, (n_stories, t_max)) - 1 # +1 because the high value is exclusive
   >>> np.unique(steps) # Verification: all steps are 1 or -1
   array([-1,  1])

  We build the walks by summing steps along the time:

  .. sourcecode:: pycon

   >>> positions = np.cumsum(steps, axis=1) # axis = 1: dimension of time
   >>> sq_distance = positions**2

  We get the mean in the axis of the stories:

  .. sourcecode:: pycon

   >>> mean_sq_distance = np.mean(sq_distance, axis=0)

  Plot the results:

  .. sourcecode:: pycon

   >>> plt.figure(figsize=(4, 3)) # doctest: +ELLIPSIS
   <matplotlib.figure.Figure object at ...>
   >>> plt.plot(t, np.sqrt(mean_sq_distance), 'g.', t, np.sqrt(t), 'y-') # doctest: +ELLIPSIS
   [<matplotlib.lines.Line2D object at ...>, <matplotlib.lines.Line2D object at ...>]
   >>> plt.xlabel(r"$t$") # doctest: +ELLIPSIS
   <matplotlib.text.Text object at ...>
   >>> plt.ylabel(r"$\sqrt{\langle (\delta x)^2 \rangle}$") # doctest: +ELLIPSIS
   <matplotlib.text.Text object at ...>
   >>> plt.tight_layout() # provide sufficient space for labels

  .. image:: auto_examples/images/sphx_glr_plot_randomwalk_001.png
     :width: 50%
     :target: auto_examples/plot_randomwalk.html
     :align: center

  We find a well-known result in physics: the RMS distance grows as the
  square root of the time!


.. arithmetic: sum/prod/mean/std

.. extrema: min/max

.. logical: all/any

.. the axis argument

.. EXE: verify if all elements in an array are equal to 1
.. EXE: verify if any elements in an array are equal to 1
.. EXE: load data with loadtxt from a file, and compute its basic statistics

.. CHA: implement mean and std using only sum()

.. _broadcasting:

广播
------------

* ``numpy`` 数组上的基本操作 (加法, etc.)是按元素的(elementwise)

* 这同样适用于相同大小(same size)的数组。

    | **尽管如此**, 不同大小的数组也可以进行上述按元素操作。
    | 如果 *NumPy* 可以变换这些数组让他们具有相同的size: 这种变换称之为 **广播(broadcasting)**.

下面的图片给出了广播的一些例子:

.. only:: latex

    .. image:: images/numpy_broadcasting.png
        :align: center

.. only:: html

    .. image:: images/numpy_broadcasting.png
        :align: center
        :width: 100%

让我们检查一下是不是这样的:

.. sourcecode:: pycon

    >>> a = np.tile(np.arange(0, 40, 10), (3, 1)).T
    >>> a
    array([[ 0,  0,  0],
           [10, 10, 10],
           [20, 20, 20],
           [30, 30, 30]])
    >>> b = np.array([0, 1, 2])
    >>> a + b
    array([[ 0,  1,  2],
           [10, 11, 12],
           [20, 21, 22],
           [30, 31, 32]])

我们已经在完全不知情的情况下使用了广播(broadcasting):

.. sourcecode:: pycon

    >>> a = np.ones((4, 5))
    >>> a[0] = 2  # 我们将维度0的数组分配给维度1的数组。 
    >>> a
    array([[ 2.,  2.,  2.,  2.,  2.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]])

一个有用的技巧:

.. sourcecode:: pycon

    >>> a = np.arange(0, 40, 10)
    >>> a.shape
    (4,)
    >>> a = a[:, np.newaxis]  # adds a new axis -> 2D array
    >>> a.shape
    (4, 1)
    >>> a
    array([[ 0],
           [10],
           [20],
           [30]])
    >>> a + b
    array([[ 0,  1,  2],
           [10, 11, 12],
           [20, 21, 22],
           [30, 31, 32]])


.. tip::

    广播似乎有点神奇，但当我们想要解决输出数据是一个比输入数据维数更大的数组时，使用它实际上是很自然的。

.. topic:: 工作案例: 广播
   :class: green

   让我们创建一个数组，它包含了66号公路沿线各城市之间的距离: 芝加哥, 斯普林菲尔德, 圣路易斯，图尔萨, 俄克拉荷马,
   阿马里洛, 圣塔菲, 阿尔伯克基, 弗洛格斯塔佛 和 洛杉矶。

   .. sourcecode:: pycon

       >>> mileposts = np.array([0, 198, 303, 736, 871, 1175, 1475, 1544,
       ...        1913, 2448])
       >>> distance_array = np.abs(mileposts - mileposts[:, np.newaxis])
       >>> distance_array
       array([[   0,  198,  303,  736,  871, 1175, 1475, 1544, 1913, 2448],
              [ 198,    0,  105,  538,  673,  977, 1277, 1346, 1715, 2250],
              [ 303,  105,    0,  433,  568,  872, 1172, 1241, 1610, 2145],
              [ 736,  538,  433,    0,  135,  439,  739,  808, 1177, 1712],
              [ 871,  673,  568,  135,    0,  304,  604,  673, 1042, 1577],
              [1175,  977,  872,  439,  304,    0,  300,  369,  738, 1273],
              [1475, 1277, 1172,  739,  604,  300,    0,   69,  438,  973],
              [1544, 1346, 1241,  808,  673,  369,   69,    0,  369,  904],
              [1913, 1715, 1610, 1177, 1042,  738,  438,  369,    0,  535],
              [2448, 2250, 2145, 1712, 1577, 1273,  973,  904,  535,    0]])


   .. image:: images/route66.png
      :align: center
      :scale: 60

许多基于网格或基于网络的问题也可以使用广播。例如，如果我们想计算10x10网格上点的起始点的距离，我们可以这样做。

.. sourcecode:: pycon

    >>> x, y = np.arange(5), np.arange(5)[:, np.newaxis]
    >>> distance = np.sqrt(x ** 2 + y ** 2)
    >>> distance
    array([[ 0.        ,  1.        ,  2.        ,  3.        ,  4.        ],
           [ 1.        ,  1.41421356,  2.23606798,  3.16227766,  4.12310563],
           [ 2.        ,  2.23606798,  2.82842712,  3.60555128,  4.47213595],
           [ 3.        ,  3.16227766,  3.60555128,  4.24264069,  5.        ],
           [ 4.        ,  4.12310563,  4.47213595,  5.        ,  5.65685425]])

或 用颜色画出来:

.. sourcecode:: pycon

    >>> plt.pcolor(distance)    # doctest: +SKIP
    >>> plt.colorbar()    # doctest: +SKIP

.. image:: auto_examples/images/sphx_glr_plot_distances_001.png
   :width: 50%
   :target: auto_examples/plot_distances.html
   :align: center


**注意** : 函数 :func:`numpy.ogrid` 可以直接创建上面的例子中的向量 x 和 y，并带有两个重要的维度。 

.. sourcecode:: pycon

    >>> x, y = np.ogrid[0:5, 0:5]
    >>> x, y
    (array([[0],
           [1],
           [2],
           [3],
           [4]]), array([[0, 1, 2, 3, 4]]))
    >>> x.shape, y.shape
    ((5, 1), (1, 5))
    >>> distance = np.sqrt(x ** 2 + y ** 2)

.. tip::

  因此, 只要当我们想要在一个网格上处理计算问题的时候， ``np.ogrid`` 是非常有用的。
  另一方面, ``np.mgrid`` 直接提供了填充满索引的矩阵，用于那些我们不能或不想使用广播的情况。

  .. sourcecode:: pycon

    >>> x, y = np.mgrid[0:4, 0:4]
    >>> x
    array([[0, 0, 0, 0],
           [1, 1, 1, 1],
           [2, 2, 2, 2],
           [3, 3, 3, 3]])
    >>> y
    array([[0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3]])

.. rules

.. some usage examples: scalars, 1-d matrix products

.. newaxis

.. EXE: add 1-d array to a scalar
.. EXE: add 1-d array to a 2-d array
.. EXE: multiply matrix from the right with a diagonal array
.. CHA: constructing grids -- meshgrid using only newaxis

.. seealso::
   
   :ref:`broadcasting_advanced`:  关于广播的讨论  :ref:`advanced_numpy` 章节。


数组形状的操作
------------------------

扁平化
..........

.. sourcecode:: pycon

    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> a.ravel()
    array([1, 2, 3, 4, 5, 6])
    >>> a.T
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> a.T.ravel()
    array([1, 4, 2, 5, 3, 6])

更高的维度: 最后一维首先被展平(ravel out)

改变形状
.........

展平操作的逆操作:

.. sourcecode:: pycon

    >>> a.shape
    (2, 3)
    >>> b = a.ravel()
    >>> b = b.reshape((2, 3))
    >>> b
    array([[1, 2, 3],
           [4, 5, 6]])

或者,

.. sourcecode:: pycon

    >>> a.reshape((2, -1))    # 未指定的值 (-1) 会被推断出来
    array([[1, 2, 3],
           [4, 5, 6]])

.. warning::

   ``ndarray.reshape`` **可能** 返回一个视图 (cf ``help(np.reshape)``)), 或者 是一个拷贝 

.. tip::

   .. sourcecode:: pycon

     >>> b[0, 0] = 99
     >>> a
     array([[99,  2,  3],
            [ 4,  5,  6]])

   要小心: reshape 可能会返回一个 copy!:

   .. sourcecode:: pycon

     >>> a = np.zeros((3, 2))
     >>> b = a.T.reshape(3*2)
     >>> b[0] = 9
     >>> a
     array([[ 0.,  0.],
            [ 0.,  0.],
            [ 0.,  0.]])

   要想充分理解这一点，你需要学习numpy数组的内存布局(memory layout of a numpy array).

添加一维
..................

使用 ``np.newaxis`` 对象索引允许我们添加一个新的数轴(axis)到一个数组
(在上面的广播一小节你已经见过了):

.. sourcecode:: pycon

    >>> z = np.array([1, 2, 3])
    >>> z
    array([1, 2, 3])

    >>> z[:, np.newaxis]
    array([[1],
           [2],
           [3]])

    >>> z[np.newaxis, :]
    array([[1, 2, 3]])



维度重排
...................

.. sourcecode:: pycon

    >>> a = np.arange(4*3*2).reshape(4, 3, 2)
    >>> a.shape
    (4, 3, 2)
    >>> a[0, 2, 1]
    5
    >>> b = a.transpose(1, 2, 0)
    >>> b.shape
    (3, 2, 4)
    >>> b[2, 1, 0]
    5

上述 transpose 只是创建了一个视图:

.. sourcecode:: pycon

    >>> b[2, 1, 0] = -1
    >>> a[0, 2, 1]
    -1

重置Size
........

数组的size可以使用 ``ndarray.resize`` 进行修改:

.. sourcecode:: pycon

    >>> a = np.arange(4)
    >>> a.resize((8,))
    >>> a
    array([0, 1, 2, 3, 0, 0, 0, 0])

然而, 被resize的数组不能在其他地方被引用:

.. sourcecode:: pycon

    >>> b = a
    >>> a.resize((4,))   # doctest: +SKIP
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: cannot resize an array that has been referenced or is
    referencing another array in this way.  Use the resize function

.. seealso: ``help(np.tensordot)``

.. resizing: how to do it, and *when* is it possible (not always!)

.. reshaping (demo using an image?)

.. dimension shuffling

.. when to use: some pre-made algorithm (e.g. in Fortran) accepts only
   1-D data, but you'd like to vectorize it

.. EXE: load data incrementally from a file, by appending to a resizing array
.. EXE: vectorize a pre-made routine that only accepts 1-D data
.. EXE: manipulating matrix direct product spaces back and forth (give an example from physics -- spin index and orbital indices)
.. EXE: shuffling dimensions when writing a general vectorized function
.. CHA: the mathematical 'vec' operation

.. topic:: **练习: 形状(Shape) 操作**
   :class: green

   * 查看 ``reshape`` 的文档, 尤其是那些需要注意的小节，包含了很多关于 copies 和 views 的信息。
   * 使用 ``flatten`` 作为 ``ravel`` 的一个替换。 那么它们的区别是啥?
     (Hint: check which one returns a view and which a copy)
   * 用 ``transpose`` 做实验学习维度重排(Dimension shuffling).

数据排序
------------

沿着某个数轴(axis)排序:

.. sourcecode:: pycon

    >>> a = np.array([[4, 3, 5], [1, 2, 1]])
    >>> b = np.sort(a, axis=1)
    >>> b
    array([[3, 4, 5],
           [1, 1, 2]])

.. note:: 分别对每一行排序!

原位(In-place)排序:

.. sourcecode:: pycon

    >>> a.sort(axis=1)
    >>> a
    array([[3, 4, 5],
           [1, 1, 2]])

使用花式索引(fancy indexing)进行排序:

.. sourcecode:: pycon

    >>> a = np.array([4, 3, 1, 2])
    >>> j = np.argsort(a)
    >>> j
    array([2, 3, 1, 0])
    >>> a[j]
    array([1, 2, 3, 4])

查找最大最小值:

.. sourcecode:: pycon

    >>> a = np.array([4, 3, 1, 2])
    >>> j_max = np.argmax(a)
    >>> j_min = np.argmin(a)
    >>> j_max, j_min
    (0, 2)


.. topic:: **练习: 排序**
   :class: green

    * 尝试原位排序和非原位排序。
    * 尝试创建不同类型的数组并对其排序。
    * 使用 ``all`` 或 ``array_equal`` 来检查结果。
    * 查看 ``np.random.shuffle`` 作为一种创建可排序输入的方法.
    * 组合使用 ``ravel``, ``sort`` 和 ``reshape`` 。
    * 查看 ``sort`` 函数 的 ``axis`` 关键字，并重做上面的排序练习。

.. topic:: 简单总结

    * 算术运算 etc. 是按元素进行操作的
    * 基本线性代数, ``.dot()``
    * 约减: ``sum(axis=1)``, ``std()``, ``all()``, ``any()``
    * 广播: ``a = np.arange(4); a[:,np.newaxis] + a[np.newaxis,:]``
    * Shape 操作: ``a.ravel()``, ``a.reshape(2, 2)``
    * 花式索引: ``a[a > 3]``, ``a[[2, 3]]``
    * 数据排序: ``.sort()``, ``np.sort``, ``np.argsort``, ``np.argmax``


总结
-------

**你需要知道什么才能开始?**

* 知道如何创建数组 : ``array``, ``arange``, ``ones``,  ``zeros``.

* 知道使用 ``array.shape`` 获得数组的shape, 然后会用切片操作获得数组的不同视图: ``array[::2]``,
  etc. 使用 ``reshape`` 调整数组的shape, 或 使用 ``ravel`` 把数组展平.

* 使用掩模(masks)获得数组的一个子集，或着 对某个子集进行修改赋值

  .. sourcecode:: pycon

     >>> a[a < 0] = 0

* 知道关于数组的一些杂七杂八的常用操作，如找最大值，平均值 (``array.max()``, ``array.mean()``)。 
  不需要记住所有事情, 但是要学会如何查找搜索文档 (online docs, ``help()``, ``lookfor()``)!!

* 对于高级使用: 掌握使用整数数组的索引，以及广播。掌握更多 NumPy 函数来处理各种各样的操作

.. topic:: **快速阅读**

   如果你想做第一次快速通过Scipy讲座来学习生态系统，你可以直接跳到下一章:
   :ref:`matplotlib` 。

   本章的其余部分没有必要跟随介绍部分的其余部分。但一定要回来完成这一章，
   并做一些更多的练习 :ref:`exercices <numpy_exercises>`.
