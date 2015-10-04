package MyLib;

import java.lang.*;
import java.io.*;
import java.awt.*;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Iterator;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;

public class FileImage {
	
	static File file;
	static BufferedImage image;
	
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
	
	public static void OpenToRead(String name) throws IOException{
		Runtime r = Runtime.getRuntime();
		r.gc();
		
		file = new File(name);
		image = ImageIO.read(file);
		
		r.gc();
	}
	
	public static int GetWidth() throws IOException{
		return image.getWidth(null);
	}
	
	public static int GetHeight() throws IOException{
		return image.getHeight(null);
	}
	
	public static int RGB(int red,int green,int blue){
		return (0xff000000)|(red<<16)|(green<<8)|blue;
	}
	
	public static void GetGrayScale(int scale[][],int width,int height) throws IOException{
		int i,j,value,red,green,blue,gray;
		for (i=0;i<width;i++)
			for (j=0;j<height;j++){
				value = image.getRGB(i,j); 
				red = (value & 0x00ff0000) >> 16;
				green = (value & 0x0000ff00) >> 8;
				blue = value & 0x000000ff;
				scale[i][j]= (red*299+green*587+blue*114)/1000;
			}
	}
	
	public static void WriteBinaryImage(int f[][],String name,int width,int height) throws IOException {
		int x,y;
		String type = GetType(name);
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		for (x=0;x<width;x++)
			for (y=0;y<height;y++)
				if (f[x][y]==1) image.setRGB(x,y,RGB(0,0,0)); else
					image.setRGB(x,y,RGB(255,255,255));
        File file = new File(name);
		ImageIO.write(image, type, file);
    }
	
	public static void WriteGrayImage(int f[][],String name,int width,int height) throws IOException {
		int x,y;
		String type = GetType(name);
		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		for (x=0;x<width;x++)
			for (y=0;y<height;y++)
				image.setRGB(x,y,RGB(f[x][y],f[x][y],f[x][y]));
        File file = new File(name);
		ImageIO.write(image, type, file);
    }
	
	public static void WriteGrayImage2(int f[][],String name,int width,int height) throws IOException {
		int x,y;
		String type = GetType(name);
		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		for (x=0;x<width;x++)
			for (y=0;y<height;y++)
				image.setRGB(x,y,RGB(255-f[x][y],255-f[x][y],255-f[x][y]));
        File file = new File(name);
		ImageIO.write(image, type, file);
    }
	
}