---
layout: post
<<<<<<< HEAD:_pre_post/2014-07-23-Selective-Search-for-Object-Recognition.md
title: "Selective Search for Object Recognition"
description:
headline:
modified: 2018-05-23
category: webdevelopment
tags: [jekyll]
imagefeature:
mathjax: true
chart:
=======
title: Selective Search for Object Recognition
description: null
headline: null
modified: {}
category:
  - object segmentation
tags:
  - jekyll
imagefeature: null
mathjax: null
chart: null
>>>>>>> 57b8c614569fd3e0424086af5978ac843ee12291:_posts/2014-07-23-Selective-Search-for-Object-Recognition.md
comments: true
featured: true
published: true
---

<<<<<<< HEAD:_pre_post/2014-07-23-Selective-Search-for-Object-Recognition.md
<!-- This paper addresses the problem of generating possible object locations for use in object recognition. -->
=======
>>>>>>> 57b8c614569fd3e0424086af5978ac843ee12291:_posts/2014-07-23-Selective-Search-for-Object-Recognition.md

Its very easy task for human to identify number of different objects in the given scene.For computers, Given an image, It "was" very hard to identify number of different objects present.  Yes its past now. With invern of RCNN, fast-RCNN, faster-RCNN, mask-RCNN and YOLO like techniques its very easy* to perform this task.  
\* huge amount of computing power is invested though.

<<<<<<< HEAD:_pre_post/2014-07-23-Selective-Search-for-Object-Recognition.md
**A lot has been changes in object recognition and  object localisation space since last 4-5 years. Thanks following papers and one man "Ross B. Girshick [http://www.rossgirshick.info/](http://www.rossgirshick.info/).**

  1. R-CNN: [https://arxiv.org/abs/1311.2524​](https://arxiv.org/abs/1311.2524​)
  2. Fast R-CNN: [https://arxiv.org/abs/1504.08083](https://arxiv.org/abs/1504.08083)​
  3. Faster R-CNN: [https://arxiv.org/abs/1506.01497​](https://arxiv.org/abs/1506.01497​)

[R-CNN](https://github.com/rbgirshick/rcnn)  (Region-based Convolutional Neural Networks) was one of the approach to identify/locate each of the object present in the given Image.

=======
**A lot has been changes in object recognition and  object localishas ation space since last 4-5 years. Thanks to following papers and one man "Ross B. Girshick"(http://www.rossgirshick.info/).**
Ross B. Girshick has greatly contributed to RCNN, fast-RCNN, faster-RCNN.

1) R-CNN: https://arxiv.org/abs/1311.2524​
2) Fast R-CNN: https://arxiv.org/abs/1504.08083​
3) Faster R-CNN: https://arxiv.org/abs/1506.01497​

>>>>>>> 57b8c614569fd3e0424086af5978ac843ee12291:_posts/2014-07-23-Selective-Search-for-Object-Recognition.md
Genuinely, R-CNN was the first step leap forward in the challenging space of object localisation. R-CNN outperforms OverFeat by a large margin on the 200-class ILSVRC2013 detection dataset.

> ["OverFeat"](https://arxiv.org/abs/1312.6229) is an approach for Integrated Recognition, Localization and Detection using Convolutional Networks  proposed by Pierre  and coworkers in 2014.

Selective Search was the one of the most important component in R-CNN. This makes understanding Selective Search more important.

Given an image Selective Search suggest locations where there is very high probability of presence of an object. Selective Search combines the strength of both an exhaustive search and segmentation.

As in exhaustive search, to capture all possible object locations. Instead of a single technique to generate possible object locations, Selective Search diversify the search and use a variety of complementary image partitioning to deal with as many image conditions as possible.

Selective Search uses image structure to guide sampling process, as segmentation does.
Well. Above sentences may be tough to understand but we will shortly understand everything. Just hold on .. keep going for while.

Before we go forward to understand Selective search we must understand challenges in locating objects in a given Image. Below given is an original example from "Selective Search" paper to highlight challenges in object localization.

Figure 1: There is a high variety of reasons that an image region forms an object. In (b) the cats can be distinguished by colour, not texture. In (c) the chameleon can be distinguished from the surrounding leaves by texture, not colour. In (d) the wheels can be part of the car because they are enclosed, not because they are similar in texture or colour. Therefore, to find objects in a structured way it is necessary to use a variety of diverse strategies. Furthermore, an image is intrinsically hierarchical as there is no single scale for which the complete table, salad bowl, and salad spoon can be found in (a).

Selective Search uses Graph search to find so called "Initial regions" as proposed in [1].

Efficient Graph-Based Image Segmentation treats each image a graph with each pixel in the image as Vertices and distance between two pixel as weight of edge.
to calculate distance between two pixel(vertices), one can use simple measure such Euclidean distance.

As per [2] The method to calculate graph from image can be formally defined as as :

> We assume that an image is given as a $p x q$ array and each pixel has an integral gray level $E [0,.X]$, $i.e.$ the whole gray scale $[0, l]$ is divided and discretized into $X + 1$ gray levels. For a given $2D$ gray-level image I, we define a weighted planar graph $G(Z) = (V,E)$, where the vertex set $V = {all pixels of I}$ and the edge set $E = \{(u, ZJ)]U, w E V$ and distunce$(u, TJ) 5 a\}$, where distunce $(u, w)$ is the Euclidean distance in terms of the number of pixels; each edge $(u, V) E E$ has a weight $W(U, V) = IQ(u) - B(V)]$, with $B(X) E [0, X]$ representing the gray level of a pixel $x E I$. Note that $G(Z)$ is a connected
graph, i.e. there exists a path between any pair of
vertices, and any vertex of $G(Z)$ has at most 8 neighbors.

Well Well Well... this is bit complicated, lets simplify a lot.
1) Lets say we have an image X. Generally image have 3 channels and it is made up of pixels as shown in image below.
2) Lets consider all pixels as all vertex of the graph.
3) calculate distance between all adjacent vertices in directed manner. For that, start form upper left corner of the image
and continue to the lower right corner of the image. While traversing in this manner you can calculate distance
between selected pixel and other three pixels 1) Pixel right to selected one, 2)Pixel bottom to selected one, and 3) Pixel diagonally bottom-right to selected one.
This all distances will create a weighted uni-directed graph.Note that grapg so created is a connected graph, i.e. there exists a path between any pair of
vertices, and any vertex of graph has at most 8 neighbors.

We are done with creating graph, next we require a minimum spanning tree of this graph. Minimum spanning tree can be described as.

>A minimum spanning tree (MST) or minimum weight spanning tree is a subset of the edges of a connected, edge-weighted (un)directed graph that connects all the vertices together, without any cycles and with the minimum possible total edge weight.

Minimum spanning tree simplifies the graph and provide that path between pixel which are connected by minimum distnace. It may
seem confusing at this point of time but you will get clear picture soon. I applied algorithm to get minimum spanning tree and I got following picture.

Original Picture

I applied various threshold and all picture are as follows:

It is clearly visible that background and foreground are perfectly separated at some threshold level.
This is what it looks like after applying MST.
Now in order to segement different region we require to break this tree in ti different parts based on
difference in weights. Selective search uses specific method called "Hierarchical Grouping Algorithm" to do this. In "selective search" paper it is given as follow:
___
**Hierarchical Grouping Algorithm**
___
* Input: (colour) image
* Output: Set of object location hypotheses L
 Obtain initial regions $R = {r1,··· ,rn}$ using [1]
**`foreach`**
  Neighbouring region pair $(ri,rj)$ do
 Calculate similarity $s(ri,rj) S = S∪s(ri,rj)$

  **`while S 6= /0 do`**
Get highest similarity $s(ri,rj) = max(S)$
Merge corresponding regions $rt = ri ∪rj$
Remove similarities regarding $ri: S = S \ s(ri,r$*)
Remove similarities regarding rj: $S = S/s(r$∗$,rj)$
Calculate similarity set St between rt and its neighbours
$S = S∪St$
$R = R∪rt$
Lets simplify this algorithm with following step:
1)  We have already generated


[1] Efficient Graph-Based Image Segmentation | https://www.cs.cornell.edu/~dph/papers/seg-ijcv.pdf
[2] 2D image segmentation using minimum spanning trees | https://doi.org/10.1016/S0262-8856(96)01105-5
