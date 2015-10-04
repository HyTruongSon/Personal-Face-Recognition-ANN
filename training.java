// Software: Training Neural Network with ORL face database
// Author: Hy Truong Son
// Major: BSc. Computer Science
// Class: 2013 - 2016
// Institution: Eotvos Lorand University
// Email: sonpascal93@gmail.com
// Website: http://people.inf.elte.hu/hytruongson/
// Final update: October 4th, 2015
// Copyright 2015 (c) Hy Truong Son. All rights reserved. Only use for academic purposes.

import java.io.*;
import java.util.Random;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;

import MyLib.*;

public class training {
    
    // Constants
	static String weightsFileName   = "weights.dat";
	static String samplesFileName   = "all-samples.dat";
	
	// Number of people
	static int nPeople          = 40;
	static int nPersonalSamples = 10;
	static int nSamples         = nPeople * nPersonalSamples; // = 400
	static int nTraining        = 320;

    // Original image size
	static int width        = 92;
	static int height       = 112;  
	
	// Resize to this size
	static int NewWidth     = 40;
	static int NewHeight    = 50;
    
    // Neural network architecture
	static int nInput   = NewWidth * NewHeight + 1;       // Input Layer, +1 as bias
	static int nHidden  = 128;                            // Hidden Layer
	static int nOutput  = nPeople;                        // Output Layer
	
	// Neural network parameters
	static int nTimes           = 32;
	static double Momentum      = 0.9;
	static int Cycle            = 512;
	static double LearningRate  = 1e-3;
	static double Epsilon       = 1e-3;
	
	static double input[]   = new double [nInput];
	static double output[]  = new double [nOutput];
	
	static int scale[][]    = new int [width][height];
	static int NewScale[][] = new int [NewWidth][NewHeight];
	
	private static void Feature(String fileName) throws IOException {
		PGM256.GetGrayScale(fileName, scale, width, height);
		Normalization.Resize(scale, NewScale, width, height, NewWidth, NewHeight);

		for (int x = 0; x < NewWidth; ++x)
			for (int y = 0; y < NewHeight; ++y){
				input[y + x * NewHeight] = (double)(NewScale[x][y]) / (double)(255);
			}
	    input[NewWidth * NewHeight] = 1.0;
	}
	
	private static void training() throws IOException {
	    System.out.println("Training the Neural Network");
	    System.out.println("Number of training samples: " + nTraining);
	    
	    // Network initialization
		NeuralNetwork ANN = new NeuralNetwork(nInput, nHidden, nOutput);
		ANN.SetMomentum(Momentum);
		ANN.SetCycle(Cycle);
		ANN.SetLearningRate(LearningRate);
		ANN.SetEpsilon(Epsilon);
		    
		// Training
		for (int t = 1; t <= nTimes; ++t) {
		    TextFileReader file = new TextFileReader();
		    file.open(samplesFileName);
		        
		    file.ReadLine();
		    for (int sample = 1; sample <= nTraining; ++sample) {
		        int subject = Integer.parseInt(file.ReadLine());
		        String imageFileName = file.ReadLine();
		            
                System.out.print("Time " + t + " - Sample " + sample + ": ");
                
                for (int i = 0; i < nOutput; ++i) {
                    output[i] = 0.0;
                }
				output[subject - 1] = 1.0;
				    
				// Feature extraction
				Feature(imageFileName);
				        
				// Back-propagation learning
				ANN.Study(input, output);
				        
				// Get the norm L2 error
				double Error = ANN.SquareError();
				        
				System.out.println(Error);
                    
                if (sample % nPeople == 0) {
                    System.out.println("Save weights to file " + weightsFileName);
                    ANN.WriteCost(weightsFileName);
                }
		    }
		        
		    file.close();
        }
		    
        System.out.println("Save weights to file " + weightsFileName);
        ANN.WriteCost(weightsFileName);
	}
	
	public static void main(String args[]) throws IOException {
	    training();
	}

}
