# Multi-Label Classification

<b>Hypothesis</b>

- Distinguish large set of grocery product images
- The three meta-categories are: material, colour, and GS1 form
- Dataset: [train-dataset.zip](https://drive.google.com/file/d/1N8kAWPheOKPUovKV1Qk42VYoGhGaXhOi/view?usp=sharing)
  - Important thing to note, in our finalized project we will use a single folder with all images, everything will be sorted within the code from the .csv file

---

<b>Sorting images to folders</b>

There is a script <code>image-folder-sort.ipynb</code>, we use it to sort our dataset. It is not necessary to use it, since we include the already sorted dataset above.

---

<b>To-do list. Last updated on 2023-01-16</b>

- [x] Reform the dataset so images match their respective classifications

- [x] Develop a custom Dataset and DataLoader 

- [x] Add data split to training and validation

- [x] Rework the training so it works with the custom dataset

- [x] Make the network's acurracy test

- [x] Discuss about the project before the exam

---

<b>Resources</b>

- [Pytorch tutorial](https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/#cnn-from-scratch)
- [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem)

