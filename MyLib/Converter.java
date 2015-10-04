package MyLib;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.io.*;
import java.awt.*;

public class Converter {

	public static String GetType(String name){
		int i,j;
		String res;
		j=0;
		for (i=0;i<name.length();i++)
			if (name.charAt(i)=='.'){
				j=i;
				break;
			}
		res="";
		for (i=j+1;i<name.length();i++) res+=name.charAt(i);
		return res;
	}

    public static void process(String FileName, String FileType1, String FileType2) throws IOException{
		String Origin = FileName+"."+FileType1;
		String Destination = FileName+"."+FileType2;
		File file= new File(Origin);
		BufferedImage image = ImageIO.read(file);
        ImageIO.write(image,FileType2,new File(Destination));
    }
	
	public static void process(String Origin, String Destination) throws IOException{
		File file= new File(Origin);
		BufferedImage image = ImageIO.read(file);
		String FileType = GetType(Destination);
        ImageIO.write(image,FileType,new File(Destination));
    }
	
}
