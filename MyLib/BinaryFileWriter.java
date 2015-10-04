package MyLib;

import java.lang.*;
import java.io.*;

public class BinaryFileWriter {
	
	BufferedOutputStream f;
	
	public void Rewrite(String name) throws IOException{
		f = new BufferedOutputStream ( new FileOutputStream(name) );
	}
	
	public void Append(String name) throws IOException{
		f = new BufferedOutputStream ( new FileOutputStream(name, true) );
	}
	
	public void write(char ch) throws IOException{
		f.write(ch);
	}
	
	public void write(int g[][],int width,int height) throws IOException{
		int x,y;
		for (y=0;y<height;y++)
			for (x=0;x<width;x++) write((char)(g[x][y]));
	}
	
	public void close() throws IOException{
		f.close();
	}
	
}