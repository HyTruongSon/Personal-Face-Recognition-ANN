package MyLib;

import java.lang.*;
import java.io.*;

public class TextFilePrinter {
	
	FileWriter file;
	PrintWriter writer;
	
	public void Rewrite(String name) throws IOException{
		file = new FileWriter(name);
        writer = new PrintWriter(file);
	}
	
	public void Append(String name) throws IOException{
		file = new FileWriter(name, true);
        writer = new PrintWriter(file);
	}
	
	public void print(double num) throws IOException{
		String s=""+num;
		writer.print(s);
	}
	
	public void println(double num) throws IOException{
		String s=""+num;
		writer.println(s);
	}
	
	public void print(int num) throws IOException{
		String s=""+num;
		writer.print(s);
	}
	
	public void println(int num) throws IOException{
		String s=""+num;
		writer.println(s);
	}
	
	public void print(String s) throws IOException{
		writer.print(s);
	}
	
	public void print() throws IOException{
		writer.print("");
	}
	
	public void println(String s) throws IOException{
		writer.println(s);
	}
	
	public void println() throws IOException{
		writer.println("");
	}
	
	public void close() throws IOException{
		writer.close();
	}
	
}