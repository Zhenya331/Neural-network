package com.company;

import java.io.File;
import java.util.ArrayList;
import java.util.Objects;

public class FileWorkMNIST {

    public static final String pathToDataset = "C:\\Users\\lorgo\\Desktop\\Study\\Магистратура\\1 курс\\Нейронные сети\\DatasetMNIST\\";

    public static final String[] pathToImagesTrain = new String[]   {"training\\0\\",
                                                                    "training\\1\\",
                                                                    "training\\2\\",
                                                                    "training\\3\\",
                                                                    "training\\4\\",
                                                                    "training\\5\\",
                                                                    "training\\6\\",
                                                                    "training\\7\\",
                                                                    "training\\8\\",
                                                                    "training\\9\\"};

    public static final String[] pathToImagesTest = new String[]   {"testing\\0\\",
                                                                    "testing\\1\\",
                                                                    "testing\\2\\",
                                                                    "testing\\3\\",
                                                                    "testing\\4\\",
                                                                    "testing\\5\\",
                                                                    "testing\\6\\",
                                                                    "testing\\7\\",
                                                                    "testing\\8\\",
                                                                    "testing\\9\\"};

    public static ArrayList<String> GetFileNames(String pathToDirectory) {
        File folder = new File(pathToDirectory);
        ArrayList<String> fileNames = new ArrayList<>();

        int count = 0;
        for (File file : Objects.requireNonNull(folder.listFiles()))
        {
            fileNames.add(file.getName());
            count++;
            if(count == 10000) {break;}
        }
        return fileNames;
    }

    public static ForDataset GetDataset(String choice) {
        ArrayList<String> fileNames = new ArrayList<>();
        ArrayList<Integer> answers = new ArrayList<>();
        int count = 0;
        if (choice.equals("train")) {
            for(int i = 0; i < pathToImagesTrain.length; i++) {
                ArrayList<String> fileNamesClass = GetFileNames(pathToDataset + pathToImagesTrain[i]);
                fileNames.addAll(fileNamesClass);
                for(String fileName: fileNamesClass) {
                    fileNames.set(count, pathToImagesTrain[i] + fileNames.get(count));
                    answers.add(i);
                    count++;
                }
            }
        }
        if (choice.equals("test")) {
            for(int i = 0; i < pathToImagesTest.length; i++) {
                ArrayList<String> fileNamesClass = GetFileNames(pathToDataset + pathToImagesTest[i]);
                fileNames.addAll(fileNamesClass);
                for(String fileName: fileNamesClass) {
                    fileNames.set(count, pathToImagesTest[i] + fileNames.get(count));
                    answers.add(i);
                    count++;
                }
            }
        }
        //for(int i = 0; i < count; i++) {System.out.println(fileNames.get(i) + ";      " + answers.get(i)); }
        return new ForDataset(fileNames, answers, count);
    }

    public static void Shuffle (ForDataset data) {
        String bufFileName;
        int bufAnswer;
        for (int i = 0; i < data.num; i++) {
            int random = (int) (i + Math.random() * (data.num - 1 - i));
            bufFileName = data.fileNames.get(random);
            bufAnswer = data.answers.get(random);
            data.fileNames.set(random, data.fileNames.get(i));
            data.fileNames.set(i, bufFileName);
            data.answers.set(random, data.answers.get(i));
            data.answers.set(i, bufAnswer);
        }
    }
}
