import os
import random

class GeneratePairs:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.
    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """

    def __init__(self, data_dir, pairs_filepath, img_ext):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = data_dir
        self.pairs_filepath = pairs_filepath
        self.img_ext = img_ext


    def generate(self):
        self._generate_matches_pairs()
        self._generate_mismatches_pairs()


    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        for name in sorted(os.listdir(self.data_dir)):
            if name == ".DS_Store":
                continue
            # print("*******************************************")
            # print(name)

            a = []
            for file in sorted(os.listdir(self.data_dir + name)):
                if file == ".DS_Store":
                    continue
                # print(a)
                a.append(file)
                # print(a)

            #
            with open(self.pairs_filepath, "a") as f:
                for i in range(3):
                    # print (a)
                    temp = random.choice(a).split("_") # This line may vary depending on how your images are named.
                    # print ("-----temp------", temp)
                    w = temp[0] #+ "_" + temp[1]
                    # print("-----W------", w)
                    # print("***************************************")
                    # print("All",random.choice(a).split("_"))
                    # print("0",random.choice(a).split("_")[0])
                    # print("1",random.choice(a).split("_")[1])

                    l = random.choice(a).split("_")[1].lstrip("0").rstrip(self.img_ext)
                    r = random.choice(a).split("_")[1].lstrip("0").rstrip(self.img_ext)
                    # print ("------L------", l)
                    # print("------R------", r)
                    f.write(temp[0] + "\t" + l + "\t" + r + "\n")


    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """
        for i, name in enumerate(os.listdir(self.data_dir)):
            if name == ".DS_Store":
                continue

            # print("i: ", i)
            # print("name: ", name)

            remaining = os.listdir(self.data_dir)

            # print("remaining before: ", remaining)

            remaining = [f_n for f_n in remaining if f_n != ".DS_Store"]

            # print("remaining after: ", remaining)

            del remaining[i] # deletes the file from the list, so that it is not chosen again

            # print("remaining final: ", remaining)


            with open(self.pairs_filepath, "a") as f:
                for i in range(3):
                    other_dir = random.choice(remaining)
                    file1 = random.choice(os.listdir(self.data_dir + name))
                    file2 = random.choice(os.listdir(self.data_dir + other_dir))
                    # print("name, other_dir: ", name, other_dir)
                    # print("Here " + name + "\t" + file1.split("_")[1].lstrip("0").rstrip(self.img_ext) + "\t")
                    # print("Here " + other_dir + "\t" + file2.split("_")[1].lstrip("0").rstrip(self.img_ext) + "\t")
                    f.write(name + "\t" + file1.split("_")[1].lstrip("0").rstrip(self.img_ext) + "\t" + other_dir + "\t" + file2.split("_")[1].lstrip("0").rstrip(self.img_ext) + "\n")
                    del remaining[remaining.index(other_dir)] # deletes the previously selected other_dir.



if __name__ == '__main__':
    data_dir = "/mnt/drive/Amir/Thesis/dataset/face-recognition/Final-face-dataset/aligned-images/test/" # do not forget to include / at the end
    pairs_filepath = "/mnt/drive/Amir/Thesis/dataset/face-recognition/Final-face-dataset/homkrun_pairs.txt"
    img_ext = ".png"
    generatePairs = GeneratePairs(data_dir, pairs_filepath, img_ext)
    generatePairs.generate()
