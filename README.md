# Multi-Label Classification

<b>Hypothesis</b>

- Distinguish large set of grocery product images
- The three meta-categories are: material, colour, and GS1 form
- Dataset: [https://bit.ly/3VcqJJP](https://bit.ly/3VcqJJP).

---

<b>Sorting images to folders</b>

To sort images into folders with desired labeling, run the <code>image-folder-sort.ipynb</code> in the main folder. In the same folder, the images must be located in the path 'images/train/' (or you can change the path in the code). 

<b>IMPORTANT!!!</b> <br>
Make sure before running, in the second code block check if there are any missing files, by uncommenting <code>print(missing_file)</code>. 
Currently, the only information missing is file's '8714100783528.jpg', so make sure it is deleted, or not in the folder.

Lastly, it seems that the information in .csv is not entirely accurate, but we can correct small mistakes like these manually, after sorting.

---

<b>To-do list. Last updated on 2022-12-10</b>

[x] Reform the dataset so images match their respective classifications
[] Make 3 models for each classifier
[] Come up with a short presentation

---

<b>Resources</b>

- [Pytorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)

