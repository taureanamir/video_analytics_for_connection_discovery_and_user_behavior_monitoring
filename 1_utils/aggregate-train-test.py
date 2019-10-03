# This scripts combines files in train and test folder into one single train folder.
#
import os
import random
import shutil

class CombineTrainTest:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.
    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """

    def __init__(self, source_path, destination_path):
        """
        Parameter source_path, is your data directory.
        Parameter destination_path, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.source_path = source_path
        self.destination_path = destination_path
        # self.num_of_files = num_of_files


    def combine(self):
        self._combine_train_test()


    def _combine_train_test(self):
        # train = "train/"
        # test = "test/"

        final_destination_train = self.destination_path
        if not os.path.exists(final_destination_train):
            os.makedirs(final_destination_train)

        for name in os.listdir(self.source_path):
            if name == ".DS_Store":
                continue

            a = []
            for file in os.listdir(self.source_path + name):
                if file == ".DS_Store":
                    continue
                a.append(file)

            # final_destination_test = self.destination_path + name
            # if not os.path.exists(final_destination_test):
            #     os.makedirs(final_destination_test)

            print("-----------AAAA------------", a)

            for i in range(len(a)):
                # print ("-----------------------", a)
                # temp = random.choice(a)
                print("# TEMP: ", a)
                print("source ", self.source_path + name)
                print("destination ", self.destination_path + name)
                # print("final_destination_test", final_destination_test)
                shutil.move(self.source_path + name + "/" + a[i], self.destination_path + name + "/" )
            #     del a[a.index(temp)]  # deletes the previously selected file.
            #
            # shutil.move(self.source_path + name, final_destination_train)


if __name__ == '__main__':
    source_path = "/home/gunner/drive/AIT/thesis/dataset/Final-face-dataset/test/"
    destination_path = "/home/gunner/drive/AIT/thesis/dataset/Final-face-dataset/train/"
    # img_ext = ".png"
    combine = CombineTrainTest(source_path, destination_path)
    combine.combine()

