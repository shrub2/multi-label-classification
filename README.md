# Multi-Label Classification

<b>Hypothesis</b>

- Distinguish large set of grocery product images
- The three meta-categories are: material, colour, and GS1 form
- Dataset: [https://bit.ly/3VcqJJP](https://bit.ly/3VcqJJP).

---

<b>Sorting images to folders</b>

To sort images into folders, execute <code>image-folder-sort.ipynb</code> in the main folder. In the same folder, the images must be located in the path 'images/train/' (or you can change the path in the code). To sort them with desired labeling: 'GS1 Form', 'Material' or 'Colour', set the <code>column</code> variable to a desired label, you should find the variable in the last code block.

<b>IMPORTANT!!!</b> <br>
Before running, in the second code block check if there are any missing files. The output should be empty if everything is OK. 
Currently, the only information missing is file's '8714100783528.jpg', so make sure it is deleted, or not in the folder.

Lastly, it seems that the information in .csv is not entirely accurate, but we can correct small mistakes like these manually, after sorting.

---

<b>To-do list. Last updated on 2022-12-10</b>

[x] Reform the dataset so images match their respective classifications

[ ] Make 3 models for each classifier

[ ] Come up with a short presentation

---

<b>Resources</b>

- [Pytorch tutorial](https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/#cnn-from-scratch)
- [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)

