Python科学计算讲座笔记 
===========================

.. only:: html

   用Python学习数字、科学和数据的文档
   --------------------------------------------------------------

.. raw html to center the title

.. raw:: html

  <style type="text/css">
    div.documentwrapper h1 {
        text-align: center;
        font-size: 280% ;
        font-weight: bold;
        margin-bottom: 4px;
    }

    div.documentwrapper h2 {
        background-color: white;
        border: none;
        font-size: 130%;
        text-align: center;
        margin-bottom: 40px;
        margin-top: 4px;
    }

    a.headerlink:after {
        content: "";
    }

    div.sidebar {
        margin-right: -20px;
        margin-top: -10px;
        border-radius: 6px;
        font-family: FontAwesome, sans-serif;
        min-width: 200pt;
    }

    div.sidebar ul {
        list-style: none;
        text-indent: -3ex;
        color: #555;
    }

    @media only screen and (max-width: 1080px) and (-webkit-min-device-pixel-ratio: 2), (max-width: 70ex)  {
        div.sidebar ul {
            text-indent: 0ex;
        }
    }

    div.sidebar li {
        margin-top: .5ex;
    }

    div.preface {
        margin-top: 20px;
    }

  </style>

.. nice layout in the toc

.. include:: tune_toc.rst

.. |pdf| unicode:: U+f1c1 .. PDF file

.. |archive| unicode:: U+f187 .. archive file

.. |github| unicode:: U+f09b  .. github logo

.. only:: html

    .. sidebar::  Download 
       
       |pdf| `PDF, 2 pages per side <./_downloads/ScipyLectures.pdf>`_

       |pdf| `PDF, 1 page per side <./_downloads/ScipyLectures-simple.pdf>`_
   
       |archive| `HTML and example files <https://github.com/scipy-lectures/scipy-lectures.github.com/zipball/master>`_
     
       |github| `Source code (github) <https://github.com/scipy-lectures/scipy-lecture-notes>`_


    关于科学Python生态系统的教程：对核心工具和技术的快速介绍。
    不同的章节分别对应一个1到2个小时的课程，
    从初学者到专家，专业水平不断提高。

    .. rst-class:: preface

        .. toctree::
            :maxdepth: 2

            preface.rst

|

.. rst-class:: tune

  .. toctree::
    :numbered: 4

    intro/index.rst
    advanced/index.rst
    packages/index.rst

|

..  
 FIXME: I need the link below to make sure the banner gets copied to the
 target directory.

.. only:: html

 .. raw:: html
 
   <div style='display: none; height=0px;'>

 :download:`ScipyLectures.pdf` :download:`ScipyLectures-simple.pdf`
 
 .. image:: themes/plusBox.png

 .. image:: images/logo.svg

 .. raw:: html
 
   </div>
   </small>


..
    >>> # For doctest on headless environments (needs to happen early)
    >>> import matplotlib
    >>> matplotlib.use('Agg')




